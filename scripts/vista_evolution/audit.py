"""Reconcile completed comparisons with their original files, without rerunning."""

import json
from pathlib import Path

from embodied_gen.harness.scene_edit.model import tool_items
from embodied_gen.harness.scene_edit.package import package_files
from embodied_gen.harness.scene_edit.records import write_json


def read(path):
    return json.loads(path.read_text())


def main():
    root = Path(
        Path('outputs/vista_evolution/latest_run.txt').read_text().strip()
    )
    manifest = read(root / 'manifest.json')
    cases = read(root / 'inputs/cases.json')
    requests = [read(p) for p in root.rglob('record.json')]
    requests = [r for r in requests if 'model' in r and 'usage' in r]
    assert len({r['directory'] for r in requests}) == len(requests)
    assert all(
        r['model'] == 'gpt-6-astra'
        and r['effort'] == 'medium'
        and not set(r.get('tool_items', [])) - {'error'}
        for r in requests
    )
    for r in requests:
        events = Path(r['directory']) / 'events.jsonl'
        if events.exists():
            parsed = []
            for line in events.read_text().splitlines():
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            assert not tool_items(parsed)
    all_results = [read(p) for p in root.rglob('result.json')]
    rendered = [
        r
        for r in all_results
        if isinstance(r.get('gpu'), dict) and 'rows' in r['gpu']
    ]
    for r in rendered:
        gpu = r['gpu']
        assert gpu['visible'] == '4'
        assert any(
            manifest['gpu_uuid'] in line and str(gpu['pid']) in line
            for line in gpu['rows']
        )
    compared = {}
    for purpose, split in [('formal', 'selection'), ('final', 'holdout')]:
        task_ids = {c['id'] for c in cases if c['split'] == split}
        for version in ['H0', 'H1']:
            eid = f'{purpose}_{split}_{version}'
            summary = read(root / 'evaluations' / eid / 'summary.json')
            assert summary['complete'] and summary['denominator'] == 24
            assert summary['turns'] == 32
            assert {e['id'] for e in summary['episodes']} == task_ids
            turns = []
            for e in summary['episodes']:
                valid = []
                for t in range(1, e['turns'] + 1):
                    d = root / 'episodes' / eid / e['id'] / f'turn_{t}'
                    attempts = [
                        read(p)
                        for p in sorted(d.glob('attempt_*/result.json'))
                    ]
                    retained = [
                        r
                        for r in attempts
                        if all(
                            q['status'] == 'ok' and q['usage'] is not None
                            for q in r['requests']
                        )
                    ]
                    assert len(retained) == 1
                    r = retained[0]
                    assert r['identity']['commit'] == summary['commit']
                    assert len(r['requests']) == 1
                    assert Path(r['output']).is_file()
                    assert len(r['gpu']['rows']) >= 1
                    if version == 'H1':
                        assert any(
                            c['skill'] == 'grounded_state'
                            and c['capability'] == 'scene.inspect@1'
                            for c in r['policy_calls']
                        )
                    valid.append(r)
                assert e['passed'] == all(
                    r['evaluation']['passed'] for r in valid
                )
                turns.extend(valid)
            assert summary['tokens'] == sum(r['known_tokens'] for r in turns)
            assert (
                abs(
                    summary['verified_s'] - sum(r['verified_s'] for r in turns)
                )
                < 1e-5
            )
            compared[eid] = turns
    for index, c in enumerate(c for c in cases if c['split'] == 'holdout'):
        order = ['H0', 'H1'] if index % 2 == 0 else ['H1', 'H0']
        first, second = (
            [
                r
                for r in compared[f'final_holdout_{v}']
                if r['identity']['case'] == c['id']
            ]
            for v in order
        )
        assert max(r['verified_mono'] for r in first) < min(
            r['start_mono'] for r in second
        )
    recommendation = read(root / 'recommended.json')
    deployed = Path('harnesses/scene_edit_vista')
    assert recommendation['id'] == 'H1'
    frozen = Path(recommendation['path'])
    assert package_files(deployed) == package_files(frozen)
    assert all(
        (deployed / f).read_bytes() == (frozen / f).read_bytes()
        for f in package_files(frozen)
    )
    result = {
        'status': 'passed',
        'complete_comparisons': {k: len(v) for k, v in compared.items()},
        'one_planner_request_per_compared_instruction': True,
        'all_revised_turns_called_registered_scene_skill': True,
        'final_episodes_interleaved_as_scheduled': True,
        'rendered_edits_with_gpu_4_process_evidence': len(rendered),
        'requests': len(requests),
        'known_tokens': sum(
            sum(
                (r.get('usage') or {}).get(k, 0)
                for k in ['input_tokens', 'output_tokens']
            )
            for r in requests
        ),
        'unknown_usage_requests': sum(r['usage'] is None for r in requests),
        'model_tool_events': 0,
        'deployed_package_matches_recommended_frozen_source': True,
    }
    assert result['requests'] <= 240 and result['known_tokens'] <= 5_000_000
    write_json(root / 'metrics/audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
