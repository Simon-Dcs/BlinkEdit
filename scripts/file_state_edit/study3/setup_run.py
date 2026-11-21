"""Freeze four-way comparisons before evaluating F2. Share immutable inputs, not outputs."""

import argparse
import copy
import csv
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

P = Path(__file__).resolve().parents[1]
BASE = P.parents[1] / 'outputs/file_state_edit'
PREV = BASE / '20261119_edit2'
ROOT = Path(os.environ.get('EDIT3_ROOT', BASE / '20261119_edit3')).resolve()
if ROOT.parent != BASE or ROOT == PREV:
    raise ValueError(
        'Choose a new sibling run under outputs/file_state_edit, not the historical baseline'
    )
sys.path.insert(0, str(P / 'study'))
from prepare import read, save


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def freeze(include_fresh):
    m = copy.deepcopy(read(PREV / 'manifest.json'))
    d = {x['chain']: x for x in read(PREV / 'setup/data/chains.json')}
    code_root = Path(m['roots']['code'])
    tasks = {
        x['case_id']: x
        for x in map(json.loads, (code_root / 'data/tasks.jsonl').open())
    }
    extra = {'edithero': [], 'code': []}
    if include_fresh:
        hosts = {
            d[k]['host_uid']
            for k in m['formal']['edithero'] + m['dev']['edithero']
        }
        for fam in ['R', 'X']:
            pool = sorted(
                k
                for k, v in d.items()
                if k.startswith(fam + '/')
                and 3 <= v['n_turns'] <= 5
                and v['host_uid'] not in hosts
                and v['license_class'] == 'CC BY / CC0'
            )
            key = pool[0]
            extra['edithero'].append(key)
            hosts.add(d[key]['host_uid'])
        sources = {
            tasks[k]['source_id']
            for k in m['formal']['code'] + m['dev']['code']
        }
        rng = random.Random(319)
        subset = tasks
        for level in ['G1', 'G4']:
            key = rng.choice(
                sorted(
                    k
                    for k in subset
                    if tasks[k]['edit_type'] == level
                    and tasks[k]['source_id'] not in sources
                )
            )
            extra['code'].append(key)
            sources.add(tasks[key]['source_id'])
    m.update(
        study='edit3',
        historical_root=str(PREV),
        comparison=m['formal'],
        fresh=extra,
        arms=['direct', 'F0', 'F1', 'F2'],
        method_roles={
            'direct': 'direct',
            'F0': 'F0',
            'F1': 'F0R',
            'F2': 'F2',
            'Fz': 'F1',
        },
        prior_arms={'F0': 'F0', 'F1': 'F0R'},
        fresh_selection='First lexicographic CC BY/CC0 chain of length 3–5 in R and X, excluding all development or previous hosts; seeded random.Random(319) sorted G1 then G4 from official full task catalog (the 312-task analysis subset has only one previously unused source), excluding previously used source IDs.',
    )
    m['formal'] = {
        label: m['comparison'][label] + extra[label] for label in extra
    }
    m['chain_info'] = {k: d[k] for k in m['formal']['edithero']}
    m['run_arms'] = {
        k: (m['arms'] if k in extra[label] else ['direct', 'F2'])
        for label in extra
        for k in m['formal'][label]
    }
    if (ROOT / 'metrics/formal-budget.json').exists() and read(
        ROOT / 'manifest.json'
    ) != m:
        raise ValueError('Cannot change population after first request')
    for sub in [
        'metrics',
        'logs',
        'setup/data',
        'setup/public_code',
        'comparisons',
    ]:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    for f in [
        'data/chains.json',
        'data/cameras.json',
        'render-rig',
        'LICENSE-DATA.md',
        'selected-asset-attribution.csv',
        'selected-code-attribution.csv',
    ]:
        link(PREV / 'setup' / f, ROOT / 'setup' / f)
    for k in m['comparison']['edithero']:
        for kind in ['states', 'renders', 'part_masks']:
            link(
                PREV / 'setup/data' / kind / k, ROOT / 'setup/data' / kind / k
            )
    for k in m['formal']['code']:
        save(
            ROOT / 'setup/public_code' / (k + '.json'),
            {j: tasks[k][j] for j in ['instruction', 'source_id']},
        )
    if include_fresh:
        for label in ['edithero', 'code']:
            if label == 'edithero':
                table = PREV / 'setup/ATTRIBUTION.csv'
                rows = [
                    v
                    for v in csv.DictReader(table.open())
                    if set(v['chains'].split()) & set(extra[label])
                ]
            else:
                ids = {tasks[k]['source_id'] for k in extra[label]}
                rows = [
                    v
                    for v in csv.DictReader(
                        (code_root / 'data/ATTRIBUTION.csv').open()
                    )
                    if v['source_id'] in ids
                ]
            if rows:
                with (ROOT / 'setup' / f'fresh-{label}-attribution.csv').open(
                    'w'
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
    save(ROOT / 'manifest.json', m)
    return m


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--fresh', action='store_true')
    a.add_argument('--fetch', action='store_true')
    args = a.parse_args()
    m = freeze(args.fresh)
    print(
        'main',
        {k: len(v) for k, v in m['comparison'].items()},
        'fresh',
        m['fresh'],
        flush=True,
    )
    if args.fetch and args.fresh:
        import prepare

        prepare.R = ROOT
        jobs = [
            ((t, k.split('/')[0]), {k})
            for t in ['states', 'renders', 'part_masks']
            for k in m['fresh']['edithero']
        ]
        with ThreadPoolExecutor(6) as pool:
            for result in pool.map(prepare.fetch_group, jobs):
                print(result, flush=True)
