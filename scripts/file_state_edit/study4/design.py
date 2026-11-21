"""Freeze the concurrent trace and fixed-state experiments before viewing any new results."""

import csv, json, os, random
from pathlib import Path

P = Path(__file__).resolve().parents[1]
BASE = P.parents[1] / 'outputs/file_state_edit'
R = Path(os.environ.get('EDIT4_ROOT', BASE / '20261120_edit4')).resolve()
OLD = BASE / '20261119_edit3'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


def link(src, dest):
    if not dest.exists():
        dest.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def freeze():
    old = read(OLD / 'manifest.json')
    ce = Path(old['roots']['code'])
    rows = {
        x['case_id']: x
        for x in map(json.loads, (ce / 'data/tasks.jsonl').open())
    }
    used = {
        rows[k]['source_id']
        for k in old['formal']['code'] + old['dev']['code']
    }
    rng = random.Random(20261120)
    fresh = []
    for kind in ['G1', 'G2', 'G3', 'G4']:
        key = rng.choice(
            sorted(
                k
                for k, t in rows.items()
                if t['source_id'] not in used and t['edit_type'] == kind
            )
        )
        fresh.append(key)
        used.add(rows[key]['source_id'])
    main = old['comparison']['code']
    keys = main + fresh
    arms = ['F1', 'F2', 'F2T']
    # Long chains are not rerun: the exact same previously produced GLB conditions both methods.
    fixed = [
        ('R/3ce5c791_s0', 1),
        ('R/3ce5c791_s0', 4),
        ('R/f76addb8_s0', 1),
        ('R/f76addb8_s0', 2),
        ('R/ba34ea4a_s2', 3),
        ('R/ba34ea4a_s2', 5),
        ('R/0def5ab1_s0', 2),
        ('R/afd4becf_s0', 3),
    ]
    tasks = [(i, rep) for i in range(8) for rep in range(3)]
    rng.shuffle(tasks)
    pairs = []
    roles = {'F1': 'F0R', 'F2': 'F2', 'F2T': 'F2T'}
    for i, rep in tasks:
        k, n = fixed[i]
        order = (
            ['full', 'select'] if (i + rep) % 2 == 0 else ['select', 'full']
        )
        b = (
            OLD / 'edithero/formal' / k / 'F2' / f'turn{n - 1:02}/outcome.json'
            if n > 1
            else None
        )
        source = (
            read(b)['artifact']
            if b
            else str(OLD / 'setup/data/states' / k / 'turn00.glb')
        )
        pair = dict(
            i=i,
            rep=rep,
            case=k,
            turn=n,
            order=order,
            prior=str(b) if b else None,
            source=source,
        )
        for method in order:
            roles[f'p{i}r{rep}-{method}'] = (
                'F2' if method == 'select' else 'F0'
            )
        pairs.append(pair)
    manifest = {
        **old,
        'study': 'edit4',
        'render_setup': 'setup/render-rig',
        'arms': arms,
        'method_roles': roles,
        'comparison': {'code': main, 'edithero': []},
        'fresh': {'code': fresh, 'edithero': []},
        'formal': {'code': keys, 'edithero': []},
        'run_arms': {k: arms for k in keys},
        'paired_trials': pairs,
        'selection': '4 distinct never previously used source programs: one official full catalog case each G1/G2/G3/G4, sorted IDs, random.Random(20261120). Eight previously exposed selector states preselected from five assets; 3 paired runs each. No winner selection.',
    }
    for sub in [
        'setup/public_code',
        'metrics',
        'logs',
        'comparisons',
        'report',
    ]:
        (R / sub).mkdir(parents=True, exist_ok=True)
    if (R / 'manifest.json').exists() and read(
        R / 'manifest.json'
    ) != manifest:
        raise RuntimeError('Fixed experiment already exists')
    write(R / 'manifest.json', manifest)
    for name in ['data', 'render-rig']:
        link(OLD / 'setup' / name, R / 'setup' / name)
    for k in keys:
        write(
            R / 'setup/public_code' / (k + '.json'),
            {j: rows[k][j] for j in ['instruction', 'source_id']},
        )
    srcids = {rows[k]['source_id'] for k in keys}
    attrs = [
        x
        for x in csv.DictReader((ce / 'data/ATTRIBUTION.csv').open())
        if x['source_id'] in srcids
    ]
    with (R / 'setup/code-attribution.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=attrs[0])
        w.writeheader()
        w.writerows(attrs)
    print(len(keys), 'programs,', len(pairs), 'pairs')


if __name__ == '__main__':
    freeze()
