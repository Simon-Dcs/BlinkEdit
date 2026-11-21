"""Resume only requests with an actual timeout. Keep every attempt in the original budget."""

import argparse, fcntl, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from design import P, R, read, write

os.environ.update(EDIT3_ROOT=str(R), BENCH_ROOT=str(R), BENCH_STAGE='formal')
sys.path[:0] = [str(P), str(P / 'study3'), str(P / 'study')]
from recover import archive, restore
from launch import job
from pairs import target, lane

M = read(R / 'manifest.json')
sys.path.insert(0, M['roots']['code'])


def timed_out(d):
    for p in d.rglob('record.json'):
        v = read(p)
        if v.get('status') != 'failed':
            continue
        if 'TimeoutExpired' in str(v.get('error')):
            return True
        events = p.with_name('events.jsonl')
        if events.exists() and '429 Too Many Requests' in events.read_text(
            errors='replace'
        ):
            return True
    return False


def held(d):
    return any(
        'Transport paused before reserving' in f.read_text()
        for f in [d / 'error.txt', d / 'review-error.txt']
        if f.exists()
    )


def save_path(d):
    for n in range(1650):
        p = R / 'diagnostics' / f'transport-{n:02}' / d.relative_to(R)
        if not p.exists():
            return p
    raise RuntimeError('No remaining attempt budget')


def code(p):
    out = p.parent
    group = out.parent
    key = group.parent.name
    arm = group.name
    v = read(p)
    with (group / 'run.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if not (v.get('error') or v.get('review_pending')) or not (
            timed_out(out) or held(out)
        ):
            return
        if v['completed'] and (timed_out(out / 'reviewer') or held(out)):
            dst = save_path(out / 'reviewer')
            if (out / 'reviewer').exists():
                archive(out / 'reviewer', dst)
            else:
                dst.mkdir(parents=True)
            write(dst / 'interrupted-outcome.json', v)
            for name in ['error.txt', 'review-error.txt']:
                if (out / name).exists():
                    (out / name).rename(dst / name)
            v.pop('error', None)
            v.update(elapsed_s=v['edit_s'], feedback_s=0, review_pending=True)
            write(p, v)
            mode = 'review'
        elif not any(
            read(x)['status'] == 'ok' for x in out.rglob('record.json')
        ):
            archive(out, save_path(out))
            mode = 'edit'
        else:
            # The standard study3 recovery restores an already delivered response and render.
            backup = save_path(p)
            backup.parent.mkdir(parents=True, exist_ok=True)
            p.rename(backup)
            mode = 'partial'
    if mode == 'partial':
        print(key, arm, restore('3dcodeedit', key, arm), flush=True)
    print(
        key,
        arm,
        mode,
        job('3dcodeedit', key, arm, False, 6, root=R, manifest=M),
        flush=True,
    )


def pairs(include_missing=False):
    tasks = []
    for pair in M['paired_trials']:
        ds = [target(pair, arm) for arm in pair['order']]
        invalid = any(timed_out(d) or held(d) for d in ds)
        split = (
            include_missing
            and sum((d / 'outcome.json').exists() for d in ds) == 1
        )
        if invalid or split:
            for d in ds:
                if d.exists():
                    archive(d, save_path(d))
            tasks.append(pair)
        elif include_missing and not all(
            (d / 'pair-score.json').exists() for d in ds
        ):
            tasks.append(pair)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda i: lane(6 + i, tasks[i::2]), range(2)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['code', 'pairs'])
    a = ap.parse_args()
    if a.kind == 'pairs':
        pairs()
    else:
        ps = [
            p
            for p in (R / '3dcodeedit/formal').glob('*/*/turn*/outcome.json')
            if (read(p).get('error') or read(p).get('review_pending'))
            and timed_out(p.parent)
        ]
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(code, ps))
