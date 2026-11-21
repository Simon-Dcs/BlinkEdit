"""Resume interrupted process work only. Reuse valid model replies and record every partial call."""

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

from setup_run import ROOT as R
from setup_run import P, read, save

sys.path.insert(0, str(P))
from embodied_gen.harness.scene_edit.records import write_json


def archive(d, root):
    bpath = R / 'metrics/formal-budget.json'
    with bpath.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        b = read(bpath)
        if root.exists():
            raise FileExistsError(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        d.rename(root)
        for i, name in enumerate(b['invocations']):
            old = Path(name)
            if old == d or d in old.parents:
                new = root / old.relative_to(d)
                b['invocations'][i] = str(new)
                # Count observed tool actions even when the CLI died before returning usage.
                if new.exists() and not list(new.rglob('record.json')):
                    ev = next(iter(new.rglob('events.jsonl')), None)
                    es = []
                    if ev:
                        for line in ev.read_text(
                            errors='replace'
                        ).splitlines():
                            try:
                                es.append(json.loads(line))
                            except ValueError:
                                continue
                    uses = [
                        z['usage']
                        for z in es
                        if z.get('type') == 'turn.completed' and z.get('usage')
                    ]
                    u = (
                        {k: sum(x.get(k, 0) for x in uses) for k in uses[0]}
                        if uses
                        else None
                    )
                    save(
                        new / 'record.json',
                        dict(
                            status='interrupted',
                            stage='editor'
                            if old.name not in ('reviewer', 'verifier')
                            else old.name,
                            usage=u,
                            tool_items=[
                                z['item']['type']
                                for z in es
                                if z.get('type') == 'item.completed'
                                and z.get('item', {}).get('type')
                                not in ('agent_message', 'reasoning', 'error')
                            ],
                            latency_s=None,
                            raw_usage=uses,
                        ),
                    )
        write_json(bpath, b)


def restore(kind, key, arm):
    import worker

    m = read(R / 'manifest.json')
    sys.path.insert(0, m['roots']['code'])
    group = R / kind / 'formal' / key / arm
    with (group / 'run.lock').open('a') as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'active'
        ds = sorted(group.glob('turn*'))
        if not ds or (ds[-1] / 'outcome.json').is_file():
            return 'nothing'
        # A detached child can outlive its queue process. Do not move that child's output.
        d = ds[-1]
        turn = int(d.name[4:])
        needle = ('worker.py', str(R), 'formal', kind, key, arm, str(turn))
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = proc.read_bytes().decode(errors='replace').split('\0')
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
            if any(
                a.endswith('/worker.py')
                and tuple(args[j + 1 : j + 7]) == needle[1:]
                for j, a in enumerate(args)
            ):
                return 'active worker'
        backup = (
            R
            / 'diagnostics/interrupted'
            / kind
            / 'formal'
            / key
            / arm
            / d.name
        )
        archive(d, backup)
        os.environ.update(
            BENCH_ROOT=str(R),
            BENCH_STAGE='formal',
            BENCH_RENDER_THREADS='4',
            BENCH_RENDER_TIMEOUT='1200',
            CUDA_VISIBLE_DEVICES='6',
        )
        orig = worker.ask
        orig_render = worker.render_program
        reused = []
        timing = {'retained_cli_s': 0.0, 'retained_render_s': 0.0}

        def ask(dest, prompt, imgs=(), stage='editor', direct_files=None):
            src = backup / dest.relative_to(d)
            rec = next(iter(src.rglob('record.json')), None)
            if (
                rec
                and read(rec).get('status') == 'ok'
                and (rec.parent / 'response.txt').is_file()
            ):
                old = (rec.parent / 'prompt.txt').read_text()
                if not (
                    old == prompt
                    or (
                        direct_files is not None
                        and old.startswith(prompt + '\nUse your file editing')
                    )
                ):
                    raise RuntimeError(
                        'Interrupted prompt differs; inspect before recovery'
                    )
                shutil.copytree(src, dest)
                timing['retained_cli_s'] += read(rec)['latency_s']
                reused.append(dest.name)
                return (rec.parent / 'response.txt').read_text()
            return orig(dest, prompt, imgs, stage, direct_files)

        def render(root, original, edited, out, tools, ce, **kw):
            old = backup / 'views'
            frame = old / 'frame.json'
            if (
                frame.is_file()
                and len(list(old.glob('*.png'))) == 12
                and edited.read_bytes() == (backup / 'edited.py').read_bytes()
            ):
                shutil.copytree(old, out)
                timing['retained_render_s'] = max(
                    0,
                    frame.stat().st_mtime
                    - (backup / 'edited.py').stat().st_mtime,
                )
                reused.append('valid_views')
                return read(frame)['modified_has_geometry']
            return orig_render(root, original, edited, out, tools, ce, **kw)

        try:
            worker.ask = ask
            worker.render_program = render
            row = worker.run(
                R, kind, key, arm, turn, 'formal', time.perf_counter()
            )
        finally:
            worker.ask = orig
            worker.render_program = orig_render
        es = sum(timing.values())
        row['elapsed_s'] += es
        if 'edit_s' in row:
            row['edit_s'] += (
                sum(
                    read(x)['latency_s']
                    for k in reused
                    if k not in ('reviewer', 'valid_views')
                    for x in (backup / k).rglob('record.json')
                )
                + timing['retained_render_s']
            )
        record = dict(
            archived=str(backup.relative_to(R)),
            reused=reused,
            **timing,
            basis='cold resumed work plus measured retained stages; discarded partial service separate',
        )
        row['recovery'] = record
        write_json(d / 'outcome.json', row)
        return ('resumed', reused, row.get('error'))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('kind')
    a.add_argument('key')
    a.add_argument('arm')
    v = a.parse_args()
    print(restore(v.kind, v.key, v.arm), flush=True)
