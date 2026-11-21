"""Project subscription transport and a separate, file-editing Codex comparator."""

import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from sandbox import PROFILE
from embodied_gen.harness.scene_edit.model import DISABLED, ModelGateway
from embodied_gen.harness.scene_edit.records import write_json

MODEL = 'gpt-6-astra'


def regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f'Not a regular file: {path}')
    return path


def ask(d, prompt, images=(), stage='editor', direct_files=None):
    from transport import admission
    with admission(Path(os.environ['BENCH_ROOT']), Path(d)):
        return _ask(d, prompt, images, stage, direct_files)


def _ask(d, prompt, images=(), stage='editor', direct_files=None):
    from budget import reserve

    if os.environ.get('BENCH_OFFLINE'):
        raise RuntimeError('Offline step cannot issue a model request')
    reserve(d)
    d = Path(d)
    d.mkdir(parents=True, exist_ok=False)
    if direct_files is None:
        gateway = ModelGateway(
            d,
            model=MODEL,
            instructions_file=Path(__file__).with_name('instructions.txt'),
        )
        limits_path = Path(os.environ['BENCH_ROOT']) / 'metrics/transport-policy.json'
        policy = json.loads(limits_path.read_text()) if limits_path.is_file() else {}
        gateway.client.timeout = policy.get('response_timeout_s', 420)
        if policy and os.environ.get('BENCH_STAGE') == 'formal':
            folder=limits_path.parent
            deadline=json.loads((folder/'formal-budget.json').read_text())['start']+json.loads((folder/'formal-limits.json').read_text())['seconds']
            gateway.client.timeout=min(gateway.client.timeout,max(1,deadline-time.time()))
        return gateway.query(prompt, [str(p) for p in images], stage=stage)
    t = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='eg-agent-') as work:
        w = Path(work)
        for name, p in direct_files.items():
            shutil.copyfile(p, w / name)
        flags = [
            'codex',
            'exec',
            '--ephemeral',
            '--json',
            '--ignore-user-config',
            '--ignore-rules',
            '--skip-git-repo-check',
            '--cd',
            work,
            '--model',
            MODEL,
            '-c',
            'model_reasoning_effort="medium"',
            '-c',
            'approval_policy="never"',
            '-c',
            'default_permissions="study"',
            '-c',
            PROFILE,
            '-c',
            'web_search="disabled"',
            '-c',
            'shell_environment_policy.inherit="none"',
            '-c',
            'shell_environment_policy.set={PATH="/usr/bin:/bin",HOME="/nonexistent"}',
        ]
        for feature in DISABLED:
            if feature not in ('shell_tool', 'unified_exec'):
                flags += ['--disable', feature]
        for i, p in enumerate(images):
            dst = w / f'image{i}.png'
            shutil.copyfile(p, dst)
            flags += ['-i', str(dst)]
        prompt += '\nUse your file editing or shell tools to write the complete answer (code, no markdown) in answer.py. Shell Python: /usr/bin/python3. Finish once written.'
        write_json(d / 'command.json', flags)
        (d / 'prompt.txt').write_text(prompt)
        record = dict(model=MODEL, stage=stage, usage=None, status='failed')
        try:
            with (
                (d / 'events.jsonl').open('w') as out,
                (d / 'stderr.txt').open('w') as err,
            ):
                r = subprocess.run(
                    flags + ['-'],
                    input=prompt,
                    text=True,
                    stdout=out,
                    stderr=err,
                    timeout=420,
                    env={
                        k: v
                        for k, v in os.environ.items()
                        if k.upper()
                        in (
                            'PATH',
                            'HOME',
                            'CODEX_HOME',
                            'HTTP_PROXY',
                            'HTTPS_PROXY',
                            'NO_PROXY',
                        )
                    },
                )
            events = [
                json.loads(line)
                for line in (d / 'events.jsonl').read_text().splitlines()
                if line.startswith('{')
            ]
            uses = [
                e['usage'] for e in events if e.get('type') == 'turn.completed'
            ]
            record['raw_usage'] = uses
            if uses:
                record['usage'] = {
                    k: sum(u.get(k, 0) for u in uses) for k in uses[0]
                }
            record['tool_items'] = [
                e['item']['type']
                for e in events
                if e.get('type') == 'item.completed'
                and e['item']['type']
                not in ('reasoning', 'agent_message', 'error')
            ]
            if r.returncode:
                raise RuntimeError(
                    'Codex CLI failed: '
                    + (d / 'stderr.txt').read_text()[-300:]
                )
            result = regular(w / 'answer.py').read_text()
            (d / 'response.txt').write_text(result)
            record['status'] = 'ok'
            return result
        finally:
            record['latency_s'] = time.perf_counter() - t
            write_json(d / 'record.json', record)
