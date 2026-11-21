"""Score only editor artifacts. GPUs 6/7 render EditHero; judge only terminal Blender attempts."""

import argparse, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from prepare import R, read

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S))
m = read(R / 'manifest.json')
sys.path.insert(0, m['roots']['code'])
from evaluate import eh, ce


def do_eh(i):
    for key in m['formal']['edithero'][i::2]:
        d = R / 'edithero/formal' / key
        data = R / 'setup/data'
        n = m['chain_info'][key]['n_turns']
        pending = [
            p
            for p in d.glob('*/turn*/outcome.json')
            if read(p).get('completed')
            and not (p.parent / 'score.json').is_file()
        ]
        if not pending:
            continue
        # eh() uses all official states to fix the shared grid; partial downloads must not alter the frame.
        if not all(
            (data / 'states' / key / f'turn{j:02}.glb').is_file()
            for j in range(n + 1)
        ):
            continue
        if not (data / 'part_masks' / key / 'masks.json').exists():
            continue
        if not all(
            (data / k / key / p.parent.name / f).exists()
            for p in pending
            for k, f in [
                ('renders', 'view3_heldout.png'),
                ('part_masks', 'edit_view3_heldout.png'),
            ]
        ):
            continue
        print('scoring', key, 'gpu', 6 + i, flush=True)
        eh(R, 'formal', key, 6 + i)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('kind', choices=['edithero', '3dcodeedit'])
    p = a.parse_args()
    os.environ.update(BENCH_ROOT=str(R), BENCH_STAGE='formal')
    if p.kind == 'edithero':
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(do_eh, range(2)))
    else:
        if not (R / 'metrics/formal-limits.json').exists():
            raise SystemExit('confirmed budget required')
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda k: ce(R, 'formal', k), m['formal']['code']))
