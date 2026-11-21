"""Serial evaluation, immutable requests and a budget shared by all model roles."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from scenarios import perturb_relocated_cup, prepare
from embodied_gen.harness.scene_edit.package import Versions
from embodied_gen.harness.scene_edit.records import append_json, write_json


def consumption(root):
    records = [
        json.loads(p.read_text()) for p in Path(root).rglob('record.json')
    ]
    records = [r for r in records if 'model' in r and 'usage' in r]
    tokens = sum(
        (r.get('usage') or {}).get('input_tokens', 0)
        + (r.get('usage') or {}).get('output_tokens', 0)
        for r in records
    )
    return dict(
        requests=len(records),
        known_tokens=tokens,
        unknown_usage=sum(r.get('usage') is None for r in records),
        records=records,
    )


def check_budget(root, reserve=0):
    c = consumption(root)
    if c['requests'] + reserve >= 240 or c['known_tokens'] >= 5000000:
        raise RuntimeError('Approved request/token limit reached')
    clock = Path(root) / 'campaign_started.json'
    if (
        clock.exists()
        and time.time() - json.loads(clock.read_text())['wall_time'] >= 14400
    ):
        raise RuntimeError('Approved four-hour experimental limit reached')
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
            'GPU 4 is in use by another compute process: ' + active
        )
    return c


def setup(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'inputs/cases.json').exists():
        raise ValueError(
            'Run inputs already exist; resume evaluate instead of setup'
        )
    for child in [
        'setup',
        'metrics',
        'inputs',
        'episodes',
        'evaluations',
        'selections',
        'comparisons',
        'development',
        'host_snapshot',
    ]:
        (root / child).mkdir(exist_ok=True)
    if not (root / 'manifest.json').exists():
        auth = subprocess.run(
            ['codex', 'login', 'status'], capture_output=True, text=True
        )
        if 'Logged in using ChatGPT' not in auth.stdout + auth.stderr:
            raise RuntimeError(
                'Sign in through codex login before creating this run'
            )
        write_json(
            root / 'manifest.json',
            dict(
                created_at=datetime.now().astimezone().isoformat(),
                status='prepared',
                gpu=4,
                model='gpt-6-astra',
                reasoning_effort='medium',
                gpu_uuid=subprocess.check_output(
                    [
                        'nvidia-smi',
                        '-i',
                        '4',
                        '--query-gpu=uuid',
                        '--format=csv,noheader',
                    ],
                    text=True,
                ).strip(),
                repository_commit=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], text=True
                ).strip(),
                limits=dict(
                    requests=240,
                    known_tokens=5000000,
                    experiment_wall_s=14400,
                    request_s=180,
                    edit_s=600,
                ),
                codex_version=subprocess.check_output(
                    ['codex', '--version'], text=True
                ).strip(),
                authentication=auth.stdout + auth.stderr,
            ),
            once=True,
        )
    cases = prepare(root)
    versions = Versions(root / 'harness_repo')
    commit, frozen = versions.freeze('harnesses/scene_edit_h0', 'Initial H0')
    write_json(
        root / 'H0.json', dict(commit=commit, path=str(frozen)), once=True
    )
    m = json.loads((root / 'manifest.json').read_text())
    m.update(
        cases=cases,
        host_protocol='scene-edit-v1',
        policy_schema=1,
        sample_points_per_object=10000,
        search_attempts_per_object=2000,
        physics=dict(
            seconds=2,
            hz=240,
            friction=[0.6, 0.5],
            contact_offset_m=0.001,
            translation_m=0.02,
            rotation_deg=5,
        ),
        formal_turns_per_version=32,
        formal_episodes_per_version=24,
        final_holdout_turns_per_version=32,
        H0_commit=commit,
    )
    write_json(root / 'manifest.json', m)
    return cases


def evaluate(
    root,
    version_id,
    policy,
    commit,
    *,
    split='selection',
    purpose='formal',
    cells=None,
    schedule_cells=None,
):
    root = Path(root).resolve()
    cases = json.loads((root / 'inputs/cases.json').read_text())
    cases = [
        c
        for c in cases
        if c['split'] == split and (cells is None or c['id'] in cells)
    ]
    full_cases = cases
    eid = f'{purpose}_{split}_{version_id}'
    d = root / 'evaluations' / eid
    d.mkdir(parents=True, exist_ok=True)
    request = dict(
        id=eid,
        version=version_id,
        commit=commit,
        purpose=purpose,
        split=split,
        cases=[c['id'] for c in cases],
        host_protocol='scene-edit-v1.1'
        if split == 'holdout'
        else 'scene-edit-v1',
    )
    write_json(d / 'request.json', request, once=True)
    catalog = json.loads((root / 'inputs/catalog.json').read_text())
    rows = []
    episodes = []
    if schedule_cells is not None:
        cases = [c for c in cases if c['id'] in schedule_cells]
    for cell in cases:
        source = Path(cell['input'])
        history = []
        steps = []
        for turn, instruction in enumerate(cell['instructions']):
            turn_dir = (
                root / 'episodes' / eid / cell['id'] / f'turn_{turn + 1}'
            )
            if cell['task'] == 'D2' and turn == 1:
                if (turn_dir / 'perturbation.json').exists():
                    source = turn_dir / 'external_update/scene.urdf'
                else:
                    source, change = perturb_relocated_cup(
                        source, turn_dir, cell
                    )
                    write_json(
                        turn_dir / 'perturbation.json', change, once=True
                    )
            recovery = root / 'setup' / 'infrastructure_recovery.json'
            permit = (
                json.loads(recovery.read_text()) if recovery.exists() else {}
            )
            key = f'{eid}/{cell["id"]}/turn_{turn + 1}'
            attempt_limit = 3 if key in permit.get('cells', []) else 2
            for attempt_index in range(1, attempt_limit + 1):
                result_path = turn_dir / f'attempt_{attempt_index}/result.json'
                if result_path.exists():
                    rec = json.loads(result_path.read_text())
                else:
                    budget = check_budget(root)
                    job = dict(
                        cell=cell,
                        turn=turn,
                        instruction=instruction,
                        input=str(source),
                        history=history,
                        policy=str(Path(policy).resolve()),
                        catalog=catalog,
                        seed=cell['seed'],
                        directory=str(result_path.parent),
                        identity=dict(
                            evaluation_id=eid,
                            version=version_id,
                            commit=commit,
                            case=cell['id'],
                            turn=turn + 1,
                            purpose=purpose,
                            split=split,
                            execution_attempt=attempt_index,
                        ),
                        video=purpose == 'formal'
                        and cell['task'] == 'T4'
                        and cell['variant'] in [0, 2],
                        max_model_requests=min(2, 240 - budget['requests']),
                    )
                    turn_dir.mkdir(parents=True, exist_ok=True)
                    job['start_mono'] = time.perf_counter()
                    job_file = turn_dir / f'job_{attempt_index}.json'
                    write_json(job_file, job)
                    with (turn_dir / f'console_{attempt_index}.log').open(
                        'w'
                    ) as log:
                        subprocess.run(
                            [
                                sys.executable,
                                'scripts/vista_evolution/worker.py',
                                str(job_file),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=600,
                            check=True,
                        )
                    rec = json.loads(result_path.read_text())
                    rec['controller_wall_s'] = (
                        time.perf_counter() - job['start_mono']
                    )
                    rec['validity'] = (
                        'invalid'
                        if any(
                            q['status'] != 'ok' or q.get('usage') is None
                            for q in rec['requests']
                        )
                        else 'valid'
                    )
                    write_json(result_path, rec)
                    append_json(root / 'metrics/turns.jsonl', rec)
                    print(
                        eid,
                        cell['id'],
                        turn + 1,
                        rec['status'],
                        'PASS'
                        if rec.get('evaluation', {}).get('passed')
                        else 'FAIL',
                        f"{rec['verified_s']:.2f}s",
                        rec['known_tokens'],
                        flush=True,
                    )
                invalid = any(
                    q['status'] != 'ok' or q.get('usage') is None
                    for q in rec['requests']
                )
                if not invalid:
                    break
                if attempt_index == attempt_limit:
                    raise RuntimeError(
                        'Infrastructure retries exhausted; preserve this incomplete evaluation'
                    )
            steps.append(rec)
            rows.append(rec)
            if rec['published']:
                source = Path(rec['output'])
                history = rec['history']
        episodes.append(
            dict(
                id=cell['id'],
                task=cell['task'],
                variant=cell['variant'],
                turns=len(steps),
                passed=all(
                    r.get('evaluation', {}).get('passed', False) for r in steps
                ),
                critical=any(
                    r.get('evaluation', {}).get('critical', False)
                    for r in steps
                ),
                tokens=sum(r['known_tokens'] for r in steps),
                verified_s=sum(r['verified_s'] for r in steps),
                session_wall_s=steps[-1]['verified_mono']
                - steps[0]['start_mono'],
            )
        )
    summary = dict(
        **request,
        complete=len(cases) == len(full_cases),
        episodes=episodes,
        turns=len(rows),
        successes=sum(e['passed'] for e in episodes),
        denominator=len(episodes),
        critical=sum(e['critical'] for e in episodes),
        tokens=sum(r['known_tokens'] for r in rows),
        verified_s=sum(r['verified_s'] for r in rows),
        llm_rpc_s=sum(r['llm_rpc_s'] for r in rows),
        per_turn=[
            dict(
                case=r['identity']['case'],
                turn=r['identity']['turn'],
                status=r['status'],
                passed=r.get('evaluation', {}).get('passed', False),
                verified_s=r['verified_s'],
                tokens=r['known_tokens'],
            )
            for r in rows
        ],
    )
    write_json(
        d
        / (
            'summary.json'
            if summary['complete']
            else 'partial_' + cases[0]['id'] + '.json'
        ),
        summary,
    )
    return summary


def evaluate_pair(root, left, right, split='holdout'):
    cases = [
        c
        for c in json.loads((Path(root) / 'inputs/cases.json').read_text())
        if c['split'] == split
    ]
    for index, cell in enumerate(cases):
        for version in [left, right] if index % 2 == 0 else [right, left]:
            evaluate(
                root,
                version['id'],
                version['path'],
                version['commit'],
                split=split,
                purpose='final',
                schedule_cells=[cell['id']],
            )
    return [
        evaluate(
            root, v['id'], v['path'], v['commit'], split=split, purpose='final'
        )
        for v in [left, right]
    ]


def select(root, before, after):
    left = {e['id']: e for e in before['episodes']}
    right = {e['id']: e for e in after['episodes']}
    if (
        left.keys() != right.keys()
        or not before['complete']
        or not after['complete']
    ):
        raise ValueError('Only complete paired matrices are comparable')
    regressions = [
        k for k in left if left[k]['passed'] and not right[k]['passed']
    ]
    accepted = False
    reason = 'tie_or_insufficient_improvement'
    if after['critical'] or regressions:
        reason = 'critical_or_previously_successful_case_regressed'
    elif after['successes'] > before['successes']:
        accepted = True
        reason = 'quality_improvement'
    elif after['successes'] == before['successes'] == before['denominator']:
        kt = 1 - after['tokens'] / before['tokens']
        tt = 1 - after['verified_s'] / before['verified_s']
        accepted = (kt >= 0.10 and tt >= -0.05) or (tt >= 0.15 and kt >= -0.05)
        if accepted:
            reason = 'efficiency_improvement'
    record = dict(
        incumbent=before['version'],
        candidate=after['version'],
        accepted=accepted,
        reason=reason,
        incumbent_successes=before['successes'],
        candidate_successes=after['successes'],
        denominator=before['denominator'],
        regressions=regressions,
        token_reduction=1 - after['tokens'] / before['tokens'],
        time_reduction=1 - after['verified_s'] / before['verified_s'],
    )
    write_json(
        Path(root) / 'selections' / f"{after['version']}.json",
        record,
        once=True,
    )
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['setup', 'evaluate'])
    p.add_argument('--root')
    p.add_argument('--version', default='H0')
    p.add_argument('--policy')
    p.add_argument('--commit')
    p.add_argument('--split', default='selection')
    p.add_argument('--purpose', default='formal')
    p.add_argument('--cells', nargs='*')
    a = p.parse_args()
    if a.action == 'setup':
        root = Path(
            a.root
            or 'outputs/vista_evolution/'
            + datetime.now().strftime('%Y%m%d_%H%M%S')
        ).resolve()
        setup(root)
        Path('outputs/vista_evolution/latest_run.txt').write_text(
            str(root) + '\n'
        )
        print(root)
        return
    root = Path(
        a.root
        or Path('outputs/vista_evolution/latest_run.txt').read_text().strip()
    )
    initial = json.loads((root / 'H0.json').read_text())
    if not (root / 'campaign_started.json').exists():
        write_json(
            root / 'campaign_started.json',
            dict(
                wall_time=time.time(),
                started_at=datetime.now().astimezone().isoformat(),
            ),
            once=True,
        )
    summary = evaluate(
        root,
        a.version,
        a.policy or initial['path'],
        a.commit or initial['commit'],
        split=a.split,
        purpose=a.purpose,
        cells=a.cells,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ['episodes', 'per_turn']
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
