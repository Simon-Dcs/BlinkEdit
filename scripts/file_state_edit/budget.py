"""Reserve before every CLI call across cold workers, including reviews and scoring."""

import fcntl
import os
import time
from pathlib import Path

from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.state_store import read


def reserve(directory):
    r = Path(os.environ['BENCH_ROOT'])
    if (r / 'metrics/transport-hold').exists(): raise RuntimeError('Transport paused before reserving a request')
    stage = os.environ['BENCH_STAGE']
    path = r / 'metrics' / f'{stage}-budget.json'
    with path.with_suffix('.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        state = (
            read(path)
            if path.exists()
            else dict(start=time.time(), invocations=[])
        )
        if stage == 'formal':
            limits = read(r / 'metrics/formal-limits.json')
        elif stage == 'dev':
            cfg = read(r / 'manifest.json')['pilot_budget']
            limits = dict(
                calls=cfg['requests'],
                tokens=cfg['tokens'],
                seconds=cfg['seconds'],
            )
        else:
            raise ValueError(f'Unknown budget: {stage}')
        n, known = len(state['invocations']), 0
        for d in state['invocations']:
            p = Path(d)
            for rec in p.rglob('record.json'):
                a = read(rec)
                u = a.get('usage') or {}
                known += u.get('input_tokens', 0) + u.get('output_tokens', 0)
                # One final response plus at most one extra response per tool item is a conservative
                # proxy for direct-agent inference calls (parallel items may share a response).
                n += len(a.get('tool_items', []))
        if (
            n >= limits['calls']
            or known >= limits['tokens']
            or time.time() - state['start'] >= limits['seconds']
        ):
            raise RuntimeError(
                f'Stage budget reached: {n} calls incl. tool-item allowance, {known} known tokens'
            )
        state['invocations'].append(str(directory))
        write_json(path, state)
