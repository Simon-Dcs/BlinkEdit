"""Dispatch once per newly terminal Blender artifact, then leave the receipt with its case."""

import concurrent.futures as futures
import os, sys, time
from prepare import R, read
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import ce

m = read(R / 'manifest.json')
sys.path.insert(0, m['roots']['code'])


def todo(k):
    state = []
    for a in m['arms']:
        ps = sorted(
            (R / '3dcodeedit/formal' / k / a).glob('turn*/outcome.json')
        )
        if not ps:
            state.append('editing')
            continue
        p = ps[-1]
        v = read(p)
        if v.get('error'):
            state.append('stopped')
        elif (v.get('accept') or v['turn'] == 1 + m['max_revisions']) and (
            p.parent / 'score.json'
        ).is_file():
            state.append('done')
        elif v.get('accept') or v['turn'] == 1 + m['max_revisions']:
            state.append('ready')
        else:
            state.append('editing')
    return state


def main():
    os.environ.update(BENCH_ROOT=str(R), BENCH_STAGE='formal')
    active = {}
    halted = set()
    with futures.ThreadPoolExecutor(4) as ex:
        while True:
            for k, fu in list(active.items()):
                if fu.done():
                    try:
                        fu.result()
                    except Exception as err:
                        print('judge error', k, repr(err), flush=True)
                        halted.add(k)
                    active.pop(k)
            waiting = False
            for k in m['formal']['code']:
                states = todo(k)
                if 'editing' in states:
                    waiting = True
                if 'ready' in states and k not in active and k not in halted:
                    active[k] = ex.submit(ce, R, 'formal', k)
            if not active and not waiting:
                break
            time.sleep(12)


if __name__ == '__main__':
    main()
