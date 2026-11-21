"""A fixed, bounded comparison, not a model-written policy search."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / 'vista_evolution')
)
from scenarios import perturb_relocated_cup
from embodied_gen.harness.scene_edit.records import write_json

VERSIONS = {
    'H1': dict(
        policy='harnesses/scene_edit_vista',
        options=dict(
            planning_mode='legacy', lean_parser=False, efficient_geometry=False
        ),
    ),
    'E1': dict(
        policy=None,
        options=dict(
            planning_mode='deterministic',
            lean_parser=True,
            efficient_geometry=True,
        ),
    ),
}


def consumption(root):
    rs = [json.loads(p.read_text()) for p in root.rglob('record.json')]
    rs = [r for r in rs if 'model' in r and 'usage' in r]
    return dict(
        requests=len(rs),
        known_tokens=sum(
            sum(
                (r.get('usage') or {}).get(k, 0)
                for k in ['input_tokens', 'output_tokens']
            )
            for r in rs
        ),
        unknown_usage=sum(r['usage'] is None for r in rs),
    )


def admission(root):
    cost = consumption(root)
    if cost['requests'] >= 80 or cost['known_tokens'] >= 2_000_000:
        raise RuntimeError('This experiment reached its request/token cap')
    started = json.loads((root / 'experiment_started.json').read_text())[
        'wall_time'
    ]
    if time.time() - started >= 7200:
        raise RuntimeError('This experiment reached its two-hour cap')
    active = subprocess.check_output(
        [
            'nvidia-smi',
            '-i',
            '4',
            '--query-compute-apps=pid',
            '--format=csv,noheader',
        ],
        text=True,
    ).strip()
    if active:
        raise RuntimeError(
            'GPU 4 is occupied; no automatic device switch: ' + active
        )


def episode(root, phase, version, cell):
    spec = VERSIONS[version]
    folder = root / 'runs' / phase / version / cell['id']
    request = dict(version=version, cell=cell, **spec)
    write_json(folder / 'request.json', request, once=True)
    source = Path(cell['input'])
    history = []
    rows = []
    for turn, instruction in enumerate(cell['instructions']):
        td = folder / f'turn_{turn + 1}'
        if cell['task'] == 'D2' and turn == 1:
            if (td / 'perturbation.json').exists():
                source = td / 'external_update/scene.urdf'
            else:
                source, change = perturb_relocated_cup(source, td, cell)
                write_json(td / 'perturbation.json', change, once=True)
        for attempt in [1, 2]:
            dest = td / f'attempt_{attempt}'
            if not (dest / 'result.json').exists():
                admission(root)
                job = dict(
                    **spec,
                    cell=cell,
                    turn=turn,
                    instruction=instruction,
                    input=str(source),
                    history=history,
                    seed=cell['seed'],
                    directory=str(dest),
                    catalog=json.loads(
                        (root / 'inputs/catalog.json').read_text()
                    ),
                    start_mono=time.perf_counter(),
                    identity=dict(
                        phase=phase,
                        version=version,
                        case=cell['id'],
                        task=cell['task'],
                        turn=turn + 1,
                        attempt=attempt,
                    ),
                )
                td.mkdir(parents=True, exist_ok=True)
                job_file = td / f'job_{attempt}.json'
                write_json(job_file, job, once=True)
                with (td / f'console_{attempt}.log').open('w') as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name('worker.py')),
                            str(job_file),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                        check=True,
                    )
            r = json.loads((dest / 'result.json').read_text())
            invalid = any(
                q['status'] != 'ok' or q.get('usage') is None
                for q in r['requests']
            )
            if not invalid:
                break
            if attempt == 2:
                raise RuntimeError(
                    'Repeated model infrastructure failure; preserve this partial comparison'
                )
        rows.append(r)
        if r['published']:
            source = Path(r['output'])
            history = r['history']
        print(
            phase,
            version,
            cell['id'],
            turn + 1,
            r['status'],
            r.get('evaluation', {}).get('passed'),
            round(r['verified_s'], 2),
            r['known_tokens'],
            flush=True,
        )
    summary = dict(
        case=cell['id'],
        task=cell['task'],
        version=version,
        phase=phase,
        passed=all(r.get('evaluation', {}).get('passed', False) for r in rows),
        critical=any(
            r.get('evaluation', {}).get('critical', False) for r in rows
        ),
        turns=len(rows),
        session_wall_s=rows[-1]['verified_mono'] - rows[0]['start_mono'],
    )
    write_json(folder / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['pilot', 'formal'])
    a = p.parse_args()
    root = Path(
        Path('outputs/edit_efficiency/latest_run.txt').read_text().strip()
    )
    if (root / f'{a.phase}_complete.json').exists():
        print('This phase is already complete; retained results are unchanged.')
        return
    cases = json.loads((root / 'inputs/cases.json').read_text())
    if a.phase == 'pilot':
        cells = [c for c in cases if c['id'] in ['D1_v0', 'D2_v0', 'T4_v0']]
        schedule = [('E1', c) for c in cells]
    else:
        cells = [c for c in cases if c['split'] == 'holdout']
        schedule = [
            (v, c)
            for i, c in enumerate(cells)
            for v in (['H1', 'E1'] if i % 2 == 0 else ['E1', 'H1'])
        ]
    write_json(
        root / f'{a.phase}_request.json',
        dict(versions=VERSIONS, schedule=[[v, c['id']] for v, c in schedule]),
        once=True,
    )
    results = [
        episode(root, a.phase, version, cell) for version, cell in schedule
    ]
    write_json(
        root / f'{a.phase}_complete.json',
        dict(episodes=results, consumption=consumption(root)),
        once=True,
    )


if __name__ == '__main__':
    main()
