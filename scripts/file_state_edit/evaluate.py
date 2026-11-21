"""Offline evaluation only: this module and its inputs are never mounted in a model sandbox."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from agents import ask
from PIL import Image
from scipy.ndimage import maximum_filter
from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.state_store import read


def meshes(path):
    s = trimesh.load(path, force='scene', skip_materials=True)
    result = {}
    for n in s.graph.nodes_geometry:
        mat, geom = s.graph[n]
        m = s.geometry[geom].copy()
        m.apply_transform(mat)
        result[n] = m
    return result


def grid(parts, low, pitch):
    result = {}
    for key, m in parts.items():
        mesh = m.copy()
        mesh.vertices = (mesh.vertices - low) / pitch - 0.5
        coords = (
            mesh.voxelized(1, method='subdivide').points.round().astype(int)
        )
        coords = coords[((coords >= 0) & (coords < 64)).all(axis=1)]
        volume = np.zeros((64, 64, 64), bool)
        volume[tuple(coords.T)] = True
        result[key] = volume
    return result


def union(parts, names=None):
    arr = np.zeros((64, 64, 64), bool)
    for name, vol in parts.items():
        if names is None or name in names:
            arr |= vol
    return arr


def iou(a, b, region):
    denom = ((a | b) & region).sum()
    return float(((a & b) & region).sum() / denom) if denom else None


def geometry_scores(gt0, gt1, old, new, initial, region, cumulative, op):
    add = gt1 & ~gt0
    rem = gt0 & ~gt1

    def recall(still, required, success):
        count = still.sum()
        if count < 8 or count < required.sum() * 0.25:
            return None
        return float(success[still].mean())

    plus = recall(add & ~old, add, new)
    minus = recall(rem & old, rem, ~new)
    val = plus if op == 'add' else minus if op == 'remove' else None
    if op == 'replace' and plus is not None and minus is not None:
        val = 2 * plus * minus / (plus + minus) if plus + minus else 0.0
    return dict(
        IF=val,
        IF_plus=plus,
        IF_minus=minus,
        CC_prev=iou(old, new, ~region),
        CC_initial=iou(initial, new, ~cumulative),
    )


def rgb(p):
    a = np.array(Image.open(p).convert('RGBA'), dtype=np.float32) / 255
    return a[:, :, :3] * a[:, :, 3:]


def fixed_render(r, key, turn, glb, out, gpu):
    m = read(r / 'manifest.json')
    tools = r.parent / 'tools'
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PXFORM_BLENDER=str(tools / 'blender-4.2.0-linux-x64/blender'),
        PXFORM_RENDER_SCRIPT=str(tools / 'blender_kit/scripts/render.py'),
    )
    out.mkdir(exist_ok=True)
    with (out / 'render.log').open('w') as log:
        subprocess.run(
            [
                sys.executable,
                str(Path(m['roots']['edithero']) / 'scripts/render_state.py'),
                '--data',
                str(r / ('setup/render-rig' if m.get('study') in ('edit2','edit3','edit4') else 'setup/data')),
                '--chain',
                key,
                '--turn',
                str(turn),
                '--glb',
                str(glb),
                '--out',
                str(out),
            ],
            env=env,
            stdout=log,
            stderr=log,
            check=True,
            timeout=480,
        )
    for i in range(4):
        if not (out / f'f{i:04}.png').is_file():
            raise RuntimeError('Incomplete held-out render')


def eh(r, stage, key, gpu):
    data = r / 'setup/data'
    m = read(r / 'manifest.json')
    chain = m['chain_info'][key]
    d = r / 'edithero' / stage / key
    start = time.perf_counter()
    mask = read(data / 'part_masks' / key / 'masks.json')['turns']
    parts = [
        meshes(data / 'states' / key / f'turn{i:02}.glb')
        for i in range(chain['n_turns'] + 1)
    ]
    bounds = np.array([v.bounds for state in parts for v in state.values()])
    lo, hi = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    side = max(hi - lo) * 64 / 63
    low = (hi + lo - side) / 2
    pitch = side / 64
    gt = [grid(s, low, pitch) for s in parts]
    totals = list(map(union, gt))
    del parts
    regions = []
    cumulative = []
    for i, v in enumerate(mask, 1):
        names = set(v['nodes'])
        if names - (gt[i - 1].keys() | gt[i].keys()):
            raise ValueError('GT node mapping is incomplete: ' + str(names))
        region = maximum_filter(
            union(gt[i - 1], names) | union(gt[i], names), 3, mode='constant'
        )
        regions.append(region)
        cumulative.append(region if i == 1 else region | cumulative[-1])
    shared = time.perf_counter() - start
    write_json(
        d / 'eval-frame.json',
        dict(low=low.tolist(), pitch=float(pitch), size=64, setup_s=shared),
    )
    for arm in m.get('run_arms',{}).get(key, m['arms']):
        old = totals[0]
        for i, op in enumerate(chain['operations'], 1):
            out = d / arm / f'turn{i:02}'
            p = out / 'outcome.json'
            if not p.exists() or not read(p)['completed']:
                break
            t = time.perf_counter()
            a = Path(read(p)['artifact'])
            new = union(grid(meshes(a), low, pitch))
            score = geometry_scores(
                totals[i - 1],
                totals[i],
                old,
                new,
                totals[0],
                regions[i - 1],
                cumulative[i - 1],
                op,
            )
            old = new
            if (out / 'score.json').exists():
                continue
            fixed_render(r, key, i, a, out / 'render', gpu)
            views = []
            for v in range(4):
                ref = f'view{v}_' + ('cond' if not v else 'heldout') + '.png'
                target = rgb(data / 'renders' / key / f'turn{i:02}' / ref)
                got = rgb(out / 'render' / f'f{v:04}.png')
                reg = maximum_filter(
                    np.array(
                        Image.open(
                            data
                            / 'part_masks'
                            / key
                            / f'turn{i:02}'
                            / ('edit_' + ref)
                        ).convert('L')
                    )
                    > 0,
                    3,
                    mode='constant',
                )
                diff = np.abs(target - got).mean(axis=2)
                views.append(
                    dict(
                        view=v,
                        pixels=int(reg.sum()),
                        edit_MAE=float(diff[reg].mean())
                        if reg.any()
                        else None,
                        unchanged_MAE=float(diff[~reg].mean()),
                    )
                )
            write_json(
                out / 'score.json',
                dict(
                    **score,
                    views=views,
                    score_s=time.perf_counter() - t,
                    gpu=gpu,
                ),
            )
    return shared


def ce(r, stage, key):
    from evaluation.prompts import VERIFIER_SYSTEM, VIEWS
    from evaluation.verify import item_prompt, parse_verdict

    manifest = read(r / 'manifest.json')
    task = next(
        t
        for t in map(
            json.loads,
            (Path(manifest['roots']['code']) / 'data/tasks.jsonl')
            .read_text()
            .splitlines(),
        )
        if t['case_id'] == key
    )
    for a in manifest.get('run_arms',{}).get(key, manifest['arms']):
        group = r / '3dcodeedit' / stage / key / a
        turns = sorted(group.glob('turn*/outcome.json'))
        if manifest.get('final_only'):
            turns = turns[-1:]
        for path in reversed(turns):
            out = path.parent
            row = read(path)
            if manifest.get('final_only') and not (row.get('accept') or row['turn'] == manifest['max_revisions'] + 1):
                continue
            t = time.perf_counter()
            if (out / 'score.json').exists() or 'error' in row:
                continue
            items = []
            pics = [
                out / 'views' / f'{s}_{v}.png'
                for v in VIEWS
                for s in ('original', 'modified')
            ]
            if row['completed']:
                for j, item in enumerate(task['checklist']):
                    target = out / 'evaluation' / f'item{j}'
                    if (target / 'verdict.json').exists():
                        items.append(read(target / 'verdict.json'))
                        continue
                    reply = ask(
                        target,
                        VERIFIER_SYSTEM
                        + '\nImages in order: '
                        + ', '.join(
                            f'{s} {v}'
                            for v in VIEWS
                            for s in ('original', 'modified')
                        )
                        + '\n'
                        + item_prompt(task, item),
                        pics,
                        stage='verifier',
                    )
                    verdict = parse_verdict(reply)
                    write_json(target / 'verdict.json', verdict)
                    items.append(verdict)
            write_json(
                out / 'score.json',
                dict(
                    all_pass=bool(
                        row['completed']
                        and all(v['answer'] == 'yes' for v in items)
                    ),
                    items=items,
                    score_s=time.perf_counter() - t,
                ),
            )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('stage')
    p.add_argument('kind', choices=['edithero', '3dcodeedit'])
    p.add_argument('--case')
    p.add_argument(
        '--budget-stage',
        choices=['dev', 'formal'],
        help='Which authorized ledger pays for outstanding checks',
    )
    a = p.parse_args()
    r = a.root.resolve()
    m = read(r / 'manifest.json')
    sys.path.insert(0, m['roots']['code'])
    os.environ.update(BENCH_ROOT=str(r), BENCH_STAGE=a.budget_stage or a.stage)
    ids = (
        [a.case]
        if a.case
        else m[a.stage]['edithero' if a.kind == 'edithero' else 'code']
    )
    if a.kind == 'edithero':
        # Exactly one renderer per physical card.
        def lane(index):
            for k in ids[index::2]:
                eh(r, a.stage, k, 6 + index)

        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lane, range(2)))
    else:
        for k in ids:
            ce(r, a.stage, k)
