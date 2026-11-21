"""The project's native Codex transport, with usage and tool visibility recorded."""

import importlib
import json
import time
from pathlib import Path

from .records import write_json

DISABLED = [
    'shell_tool',
    'unified_exec',
    'plugins',
    'remote_plugin',
    'apps',
    'browser_use',
    'browser_use_external',
    'browser_use_full_cdp_access',
    'computer_use',
    'image_generation',
    'multi_agent',
    'multi_agent_v2',
    'skill_search',
]


def tool_items(events):
    """Transport diagnostics are notifications, not executable model tools."""
    return [
        e['item']['type']
        for e in events
        if e.get('type') == 'item.completed'
        and e['item']['type'] not in ['reasoning', 'agent_message', 'error']
    ]


class ModelGateway:
    def __init__(
        self,
        directory,
        *,
        model='gpt-6-astra',
        effort='medium',
        identity=None,
        instructions_file=None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model, self.effort = model, effort
        self.identity = identity or {}
        self.instructions_file = (
            Path(instructions_file).resolve() if instructions_file else None
        )
        if self.instructions_file:
            self.instructions_file.read_text()
        self.records = []
        self.gc = importlib.import_module('embodied_gen.utils.gpt_clients')
        self.client = self.gc.GPTclient(
            None,
            None,
            model_name=model,
            provider='codex',
            check_connection=False,
            timeout=180,
        )

    def query(self, prompt, images=None, stage='planner'):
        original = self.gc.subprocess
        gateway = self
        d = self.directory / f'request_{len(self.records):02d}'
        d.mkdir()

        class Proxy:
            def __getattr__(self, key):
                return getattr(original, key)

            def run(self, command, **kw):
                command = list(command)
                assert command[1] == 'exec'
                flags = ['--json', '--config', 'web_search="disabled"']
                if gateway.instructions_file:
                    flags += [
                        '--config',
                        'model_instructions_file='
                        + json.dumps(str(gateway.instructions_file)),
                    ]
                for feature in DISABLED:
                    flags += ['--disable', feature]
                command[2:2] = flags
                write_json(d / 'command.json', command)
                (d / 'prompt.txt').write_text(kw['input'])
                record = dict(
                    **gateway.identity,
                    model=gateway.model,
                    effort=gateway.effort,
                    stage=stage,
                    status='failed',
                    usage=None,
                    directory=str(d),
                    instructions_file=str(gateway.instructions_file)
                    if gateway.instructions_file
                    else None,
                )
                t = time.perf_counter()
                try:
                    result = original.run(command, **kw)
                    (d / 'events.jsonl').write_text(result.stdout)
                    (d / 'stderr.txt').write_text(result.stderr)
                    events = []
                    for line in result.stdout.splitlines():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                    uses = [
                        e['usage']
                        for e in events
                        if e.get('type') == 'turn.completed'
                    ]
                    record['raw_usage'] = uses
                    if uses:
                        record['usage'] = {
                            k: sum(u.get(k, 0) for u in uses) for k in uses[0]
                        }
                    record['tool_items'] = tool_items(events)
                    record['transport_diagnostics'] = [
                        e
                        for e in events
                        if e.get('type') == 'error'
                        or (
                            e.get('type') == 'item.completed'
                            and e['item']['type'] == 'error'
                        )
                    ]
                    answer = Path(
                        command[command.index('--output-last-message') + 1]
                    )
                    if answer.exists():
                        (d / 'response.txt').write_text(answer.read_text())
                    record.update(
                        returncode=result.returncode,
                        status='ok'
                        if result.returncode == 0 and not record['tool_items']
                        else 'failed',
                    )
                    if record['tool_items']:
                        raise RuntimeError(
                            'Model tool boundary violated: '
                            + str(record['tool_items'])
                        )
                    return result
                except Exception as exc:
                    # subprocess.run keeps partial streams on TimeoutExpired. Preserve
                    # the CLI's own diagnostics, as for a normal nonzero exit.
                    if isinstance(exc, original.TimeoutExpired):
                        for name, partial in [('events.jsonl', exc.stdout), ('stderr.txt', exc.stderr)]:
                            if partial:
                                (d / name).write_bytes(partial if isinstance(partial, bytes) else partial.encode())
                    record['error'] = repr(exc)
                    raise
                finally:
                    record['latency_s'] = time.perf_counter() - t
                    gateway.records.append(record)
                    write_json(d / 'record.json', record)

        self.client._codex_run = Proxy().run
        try:
            reply = self.client.query(
                prompt,
                image_base64=images,
                params={
                    'model': self.model,
                    'model_reasoning_effort': self.effort,
                },
            )
        finally:
            del self.client._codex_run
        if reply is None:
            raise RuntimeError(
                'Native Codex request failed; see request evidence'
            )
        return reply
