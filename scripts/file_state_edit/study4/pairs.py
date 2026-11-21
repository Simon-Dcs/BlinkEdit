"""Fixed incoming assets, balanced method order; all evaluation follows both timed edits."""

import os, sys, time, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from design import P, R, OLD, read, write

sys.path.insert(0, str(P))
from evaluate import fixed_render
from embodied_gen.harness.asset_edit import gltf


def arm(pair, method):
    return f"p{pair['i']}r{pair['rep']}-{method}"


def target(pair, method):
    return (
        R
        / 'edithero/formal'
        / pair['case']
        / arm(pair, method)
        / f"turn{pair['turn']:02}"
    )


def score(pair, method, gpu):
    case = pair['case']
    n = pair['turn']
    out = target(pair, method)
    if (out / 'pair-score.json').is_file():
        return
    row = read(out / 'outcome.json')
    truth = read(R / 'setup/data/part_masks' / case / 'masks.json')['turns'][
        n - 1
    ]['nodes']
    src = Path(pair['source'])
    gs, _ = gltf.header(src)
    before = gltf.observe(gs)[0]['nodes']
    available = set(truth) <= set(v['name'] for v in before.values())
    s = dict(expected=truth, available=available, exact=False)
    if row['completed']:
        dest = Path(row['artifact'])
        gd, _ = gltf.header(dest)
        after = gltf.observe(gd)[0]['nodes']
        removed = set(before) - set(after)
        sibling_unchanged = all(
            k in before
            and all(
                v == before[k].get(j)
                for j, v in node.items()
                if j != 'stable_id'
            )
            for k, node in after.items()
        )
        s.update(
            removed=sorted(before[k]['name'] for k in removed),
            unchanged_others=sibling_unchanged,
        )
        s['exact'] = (
            available and s['removed'] == sorted(truth) and sibling_unchanged
        )
        fixed_render(R, case, n, dest, out / 'render', gpu)
        from evaluate import rgb
        import numpy as np
        from PIL import Image
        from scipy.ndimage import maximum_filter

        dif = []
        for view in range(1, 4):
            suffix = f'view{view}_heldout.png'
            t = f'turn{n:02}'
            mask = maximum_filter(
                np.asarray(
                    Image.open(
                        R
                        / 'setup/data/part_masks'
                        / case
                        / t
                        / ('edit_' + suffix)
                    ).convert('L')
                )
                > 0,
                3,
            )
            if mask.any():
                dif.append(
                    float(
                        np.abs(
                            rgb(out / 'render' / f'f{view:04}.png')
                            - rgb(R / 'setup/data/renders' / case / t / suffix)
                        )[mask].mean()
                    )
                )
        s['heldout_mae'] = float(np.mean(dif)) if dif else None
    write(out / 'pair-score.json', s)


def lane(gpu, pairs):
    for pair in pairs:
        for method in pair['order']:
            out = target(pair, method)
            out.parent.mkdir(parents=True, exist_ok=True)
            if (out / 'outcome.json').is_file():
                continue
            if pair['prior']:
                z = read(pair['prior'])
                z['conditioning_only'] = True
                write(
                    out.parent / f"turn{pair['turn'] - 1:02}" / 'outcome.json',
                    z,
                )
            env = dict(
                os.environ,
                BENCH_ROOT=str(R),
                BENCH_STAGE='formal',
                CUDA_VISIBLE_DEVICES=str(gpu),
            )
            with (out.parent / 'worker.log').open('a') as f:
                env['BENCH_MONO'] = str(time.perf_counter())
                subprocess.run(
                    [
                        sys.executable,
                        str(P / 'worker.py'),
                        str(R),
                        'formal',
                        'edithero',
                        pair['case'],
                        arm(pair, method),
                        str(pair['turn']),
                    ],
                    env=env,
                    stdout=f,
                    stderr=f,
                    check=True,
                )
        for method in pair['order']:
            score(pair, method, gpu)
        print('done', pair['i'], pair['rep'], flush=True)


if __name__ == '__main__':
    pairs = read(R / 'manifest.json')['paired_trials']
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda i: lane(6 + i, pairs[i::2]), range(2)))
