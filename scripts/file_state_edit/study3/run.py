"""Run independent chains/cases. Scoring is separate; never reset failed chains to references."""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from setup_run import ROOT, P, read

sys.path[:0] = [str(P), str(P / 'study')]
from evaluate import ce, eh
from launch import job


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('operation', choices=['edit', 'score'])
    cli.add_argument('kind', choices=['edithero', '3dcodeedit'])
    cli.add_argument('--arms', nargs='+')
    cli.add_argument('--workers', type=int, default=4)
    cli.add_argument('--only', choices=['main', 'fresh'])
    args = cli.parse_args()
    m = read(ROOT / 'manifest.json')
    if not (ROOT / 'metrics/formal-limits.json').exists() and (
        args.operation == 'edit' or args.kind == '3dcodeedit'
    ):
        raise SystemExit('Edit3 budget authorization required')
    os.environ.update(BENCH_ROOT=str(ROOT), BENCH_STAGE='formal')
    tag = 'code' if args.kind == '3dcodeedit' else 'edithero'
    keys = m[
        'formal'
        if not args.only
        else 'fresh'
        if args.only == 'fresh'
        else 'comparison'
    ][tag]
    sys.path.insert(0, m['roots']['code'])
    if args.operation == 'edit':
        tasks = []
        for i, key in enumerate(keys):
            arms = m['run_arms'][key]
            for arm in arms[i % len(arms) :] + arms[: i % len(arms)]:
                if args.arms and arm not in args.arms:
                    continue
                tasks.append((args.kind, key, arm, False, 6 + i % 2))

        def one(task):
            return job(*task, root=ROOT, manifest=m)
    elif args.kind == 'edithero':
        # One rendering service per physical GPU; checks successful products only.
        tasks = [(6 + i, keys[i::2]) for i in (0, 1)]

        def one(task):
            gpu, ks = task
            for k in ks:
                p = ROOT / 'edithero/formal' / k
                if any(
                    read(z).get('completed')
                    and not (z.parent / 'score.json').is_file()
                    for z in p.glob('*/turn*/outcome.json')
                ):
                    eh(ROOT, 'formal', k, gpu)
    else:
        tasks = keys

        def one(key):
            return ce(ROOT, 'formal', key)

    errors = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(one, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                print(t, fut.result(), flush=True)
            except Exception as e:
                print('ERROR', t, repr(e), flush=True)
                errors.append(str(e))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
