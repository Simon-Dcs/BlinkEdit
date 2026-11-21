"""Grade a chain/case once all its newly run methods have stopped; one owner per GPU."""

import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from setup_run import ROOT, P, read

sys.path.insert(0, str(P))
from evaluate import ce, eh


def stopped(kind, key, arm, m):
    ps = sorted(
        (ROOT / kind / 'formal' / key / arm).glob('turn*/outcome.json')
    )
    if not ps:
        return False
    z = read(ps[-1])
    if z.get('error'):
        return True
    if kind == 'edithero':
        return (
            not z['completed'] or z['turn'] == m['chain_info'][key]['n_turns']
        )
    return z.get('accept') or z['turn'] >= 1 + m['max_revisions']


def main():
    m = read(ROOT / 'manifest.json')
    sys.path.insert(0, m['roots']['code'])
    os.environ.update(BENCH_ROOT=str(ROOT), BENCH_STAGE='formal')
    groups = [
        ('edithero', m['formal']['edithero'][i::2], 6 + i) for i in range(2)
    ]
    groups += [
        ('3dcodeedit', m['formal']['code'][i::4], None) for i in range(4)
    ]

    def lane(group):
        kind, keys, gpu = group
        todo = set(keys)
        while todo:
            for k in keys:
                if k in todo and all(
                    stopped(kind, k, a, m) for a in m['run_arms'][k]
                ):
                    try:
                        if kind == 'edithero':
                            eh(ROOT, 'formal', k, gpu)
                        else:
                            ce(ROOT, 'formal', k)
                        print('graded', kind, k, flush=True)
                    except Exception:
                        traceback.print_exc()
                    todo.remove(k)
            if todo:
                time.sleep(5)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lane, groups))


if __name__ == '__main__':
    main()
