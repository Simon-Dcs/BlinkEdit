"""Finish interrupted, ungraded stages after the ordinary runners have drained."""

import argparse, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from resume_transport import (
    R,
    M,
    P,
    code,
    pairs,
    timed_out,
    held,
    save_path,
    archive,
    read,
)
from evaluate import ce
from launch import job


def finish_edit(key):
    arms = M['run_arms'][key]
    i = M['formal']['code'].index(key) % len(arms)
    for a in arms[i:] + arms[:i]:
        ds = sorted(
            (R / '3dcodeedit/formal' / key / a).glob('turn*/outcome.json')
        )
        row = read(ds[-1]) if ds else {}
        if row.get('accept') or row.get('turn', 0) == 9:
            continue
        if (
            ds
            and (row.get('error') or row.get('review_pending'))
            and (timed_out(ds[-1].parent) or held(ds[-1].parent))
        ):
            code(ds[-1])
        elif not row.get('error'):
            print(
                key,
                a,
                job('3dcodeedit', key, a, False, 6, root=R, manifest=M),
                flush=True,
            )
        else:
            print('inspect non-timeout', key, a, row.get('error'), flush=True)


def finish_grade(key):
    base = R / '3dcodeedit/formal' / key
    for d in base.glob('*/turn*/evaluation/item*'):
        if not (d / 'verdict.json').exists() and timed_out(d):
            archive(d, save_path(d))
    ce(R, 'formal', key)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['edit', 'grade', 'pairs'])
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if args.stage == 'pairs':
        pairs(include_missing=True)
    else:
        with ThreadPoolExecutor(args.workers) as ex:
            list(
                ex.map(
                    finish_edit if args.stage == 'edit' else finish_grade,
                    M['formal']['code'],
                )
            )
