"""Resume frozen cases; a fresh worker for each arm and turn, interleaved by turn."""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from embodied_gen.harness.scene_edit.state_store import read


def case(args, kind, key, index):
    root = args.root.resolve()
    m = read(root / 'manifest.json')
    arms = ['F0', 'F1', 'direct']
    turns = (
        m['chain_info'][key]['n_turns']
        if kind == 'edithero'
        else args.revisions + 1
    )
    stopped = set()
    for turn in range(1, turns + 1):
        shift = (turn + index) % 3
        for arm in arms[shift:] + arms[:shift]:
            if arm in stopped:
                continue
            d = root / kind / args.stage / key / arm / f'turn{turn:02}'
            env = dict(
                os.environ,
                BENCH_ROOT=str(root),
                BENCH_STAGE=args.stage,
                CUDA_VISIBLE_DEVICES=str(6 + index % 2),
            )
            if not d.exists():
                env['BENCH_MONO'] = str(time.perf_counter())
                with (
                    root
                    / 'setup'
                    / f'{args.stage}-{key.replace("/", "_")}.log'
                ).open('a') as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name('worker.py')),
                            str(root),
                            args.stage,
                            kind,
                            key,
                            arm,
                            str(turn),
                        ],
                        env=env,
                        stdout=log,
                        stderr=log,
                        check=True,
                    )
            elif not (d / 'outcome.json').exists():
                print('active or interrupted (no result yet)', d, flush=True)
                return
            row = read(d / 'outcome.json')
            if 'error' in row:
                if 'budget reached' in row['error']:
                    return
                if 'Codex' in row['error']:
                    raise RuntimeError(row['error'])
                stopped.add(arm)
            elif (kind == 'edithero' and not row['completed']) or (
                kind == '3dcodeedit' and row.get('accept')
            ):
                stopped.add(arm)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('stage', choices=['dev', 'formal'])
    p.add_argument('--kind', choices=['edithero', '3dcodeedit'])
    p.add_argument('--revisions', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    m = read(args.root / 'manifest.json')
    jobs = [
        (kind, k, i)
        for kind, typ in [('edithero', 'edithero'), ('3dcodeedit', 'code')]
        if args.kind in [None, kind]
        for i, k in enumerate(m[args.stage][typ])
    ]
    with ThreadPoolExecutor(args.workers) as pool:
        for fut in [pool.submit(case, args, *job) for job in jobs]:
            fut.result()
