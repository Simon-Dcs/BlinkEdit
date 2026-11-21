"""Summarize retained actual executions and produce scientific figures."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from embodied_gen.harness.scene_edit.model import tool_items
from embodied_gen.harness.scene_edit.records import write_json


def csv_file(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    root = Path(
        Path('outputs/edit_efficiency/latest_run.txt').read_text().strip()
    )
    turns = []
    for p in (root / 'runs').glob('*/*/*/turn_*/attempt_*/result.json'):
        r = json.loads(p.read_text())
        r['directory'] = str(p.parent)
        turns.append(r)
    turns.sort(key=lambda r: r['start_mono'])
    requests = [json.loads(p.read_text()) for p in root.rglob('record.json')]
    requests = [r for r in requests if 'model' in r and 'usage' in r]
    rows = []
    for r in turns:
        rows.append(
            dict(
                **r['identity'],
                instruction=r['instruction'],
                status=r['status'],
                passed=r.get('evaluation', {}).get('passed', False),
                critical=r.get('evaluation', {}).get('critical', False),
                valid=all(
                    q['status'] == 'ok' and q['usage'] is not None
                    for q in r['requests']
                ),
                known_tokens=r['known_tokens'],
                unknown_usage=r['unknown_usage_requests'],
                verified_s=r['verified_s'],
                verified_time_kind=r.get('verified_time_kind', 'measured'),
                model_s=r['llm_rpc_s'],
                local_s=r['non_llm_s'],
                startup_s=r['startup_s'],
                input_tokens=sum(
                    (q['usage'] or {}).get('input_tokens', 0)
                    for q in r['requests']
                ),
                output_tokens=sum(
                    (q['usage'] or {}).get('output_tokens', 0)
                    for q in r['requests']
                ),
                prompt_characters=len(
                    (Path(r['directory']) / 'planner_input.txt').read_text()
                )
                if (Path(r['directory']) / 'planner_input.txt').exists()
                else None,
                **r['stages'],
                directory=r['directory'],
            )
        )
    csv_file(root / 'metrics/turns.csv', rows)
    csv_file(
        root / 'metrics/requests.csv',
        [
            {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            | (r['usage'] or {})
            for r in requests
        ],
    )
    summary = dict(
        requests=len(requests),
        known_tokens=sum(
            sum(
                (r['usage'] or {}).get(k, 0)
                for k in ['input_tokens', 'output_tokens']
            )
            for r in requests
        ),
        unknown_usage=sum(r['usage'] is None for r in requests),
        interrupted_requests=sum(
            r['status'] == 'interrupted' for r in requests
        ),
        versions={},
        phases={},
    )
    for phase in ['setup', 'pilot', 'formal']:
        qs = [r for r in requests if '/' + phase + '/' in r['directory']]
        summary['phases'][phase] = dict(
            requests=len(qs),
            known_tokens=sum(
                sum(
                    (r['usage'] or {}).get(k, 0)
                    for k in ['input_tokens', 'output_tokens']
                )
                for r in qs
            ),
            unknown_usage=sum(r['usage'] is None for r in qs),
        )
    complete = (root / 'formal_complete.json').exists()
    pairs = []
    for v in ['H1', 'E1']:
        episodes = [
            json.loads(p.read_text())
            for p in (root / 'runs/formal' / v).glob('*/summary.json')
        ]
        vals = [
            r for r in rows if r['phase'] == 'formal' and r['version'] == v
        ]
        completed = [r for r in vals if r['valid']]
        if not episodes:
            continue
        edits = [
            e for e in episodes if e['task'] not in ['B1', 'B2', 'B3', 'D3']
        ]
        boundaries = [
            e for e in episodes if e['task'] in ['B1', 'B2', 'B3', 'D3']
        ]
        n = sum(e['turns'] for e in episodes)
        summary['versions'][v] = dict(
            episodes=episodes,
            completed_episodes=len(episodes),
            turns=n,
            successes=sum(e['passed'] for e in episodes),
            critical=sum(e['critical'] for e in episodes),
            editing_successes=sum(e['passed'] for e in edits),
            editing_tasks=len(edits),
            boundary_successes=sum(e['passed'] for e in boundaries),
            boundary_tasks=len(boundaries),
            attempts=len(vals),
            known_tokens=sum(r['known_tokens'] for r in vals),
            unknown_usage=sum(r['unknown_usage'] for r in vals),
            interrupted_time_upper_bound_s=sum(
                r['verified_s']
                for r in vals
                if r['verified_time_kind'] != 'measured'
            ),
            verified_s=sum(r['verified_s'] for r in vals),
            model_s=sum(r['model_s'] for r in vals),
            local_s=sum(r['local_s'] for r in vals),
            input_tokens=sum(r['input_tokens'] for r in vals),
            output_tokens=sum(r['output_tokens'] for r in vals),
            completed_attempt_only_s=sum(r['verified_s'] for r in completed),
            mean_prompt_characters=sum(
                r['prompt_characters'] or 0 for r in completed
            )
            / max(1, len(completed)),
        )
    summary['complete'] = complete
    if complete:
        left = summary['versions']['H1']
        right = summary['versions']['E1']
        assert left['completed_episodes'] == right['completed_episodes'] == 24
        assert left['turns'] == right['turns'] == 32
        le = {e['case']: e for e in left['episodes']}
        re = {e['case']: e for e in right['episodes']}
        assert le.keys() == re.keys()
        regressions = [
            k for k in le if le[k]['passed'] and not re[k]['passed']
        ]
        token_change = right['known_tokens'] / left['known_tokens'] - 1
        time_change = right['verified_s'] / left['verified_s'] - 1
        quality = (
            not regressions
            and right['critical'] == 0
            and right['successes'] == 24
        )
        efficiency = (token_change <= -0.1 and time_change <= 0.05) or (
            time_change <= -0.15 and token_change <= 0.05
        )
        # Unknown usage cannot establish a complete token reduction.
        token_evidence_complete = (
            left['unknown_usage'] == right['unknown_usage'] == 0
        )
        accepted = quality and efficiency and token_evidence_complete
        summary.update(
            regressions=regressions,
            token_change=token_change,
            time_change=time_change,
            local_change=right['local_s'] / left['local_s'] - 1,
            model_change=right['model_s'] / left['model_s'] - 1,
            quality_passed=quality,
            efficiency_passed=efficiency,
            token_evidence_complete=token_evidence_complete,
            recommended='E1' if accepted else 'H1',
        )
        for key in le:
            pair = dict(
                case=key,
                H1_passed=le[key]['passed'],
                E1_passed=re[key]['passed'],
            )
            for v in ['H1', 'E1']:
                vals = [
                    r
                    for r in rows
                    if r['phase'] == 'formal'
                    and r['version'] == v
                    and r['case'] == key
                ]
                for out, k in [
                    ('seconds', 'verified_s'),
                    ('tokens', 'known_tokens'),
                ]:
                    pair[v + '_' + out] = sum(r[k] for r in vals)
            pairs.append(pair)
        csv_file(root / 'metrics/pairs.csv', pairs)
        write_json(
            root / 'recommendation.json',
            {
                k: summary[k]
                for k in [
                    'recommended',
                    'quality_passed',
                    'efficiency_passed',
                    'regressions',
                    'token_change',
                    'time_change',
                    'token_evidence_complete',
                ]
            },
        )
    write_json(root / 'metrics/summary.json', summary)
    # Audit the effective inputs and measurements, not a rerun of the experiment.
    assert summary['requests'] <= 80 and summary['known_tokens'] <= 2_000_000
    for q in requests:
        assert (
            q['model'] == 'gpt-6-astra'
            and q['effort'] == 'medium'
            and not q.get('tool_items')
        )
        events = Path(q['directory']) / 'events.jsonl'
        if events.exists():
            parsed = []
            for line in events.read_text().splitlines():
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            assert not tool_items(parsed)
    rendered = [r for r in turns if r.get('gpu')]
    for r in rendered:
        assert r['gpu']['visible'] == '4'
        assert any(
            'GPU-b692220b-abe6-f77f-7740-0ad29a1150ce' in line
            and str(r['gpu']['pid']) in line
            for line in r['gpu']['rows']
        )
    for r in turns:
        assert Path(r['output']).is_file()
        assert r['known_tokens'] == sum(
            sum(
                (q['usage'] or {}).get(k, 0)
                for k in ['input_tokens', 'output_tokens']
            )
            for q in r['requests']
        )
    frozen = root / 'source_frozen/embodied_gen/harness/scene_edit'
    for p in frozen.iterdir():
        if p.suffix in ['.py', '.txt']:
            assert (
                p.read_bytes()
                == (
                    Path('embodied_gen/harness/scene_edit') / p.name
                ).read_bytes()
            ), p.name
    if complete:
        cases = [
            c
            for c in json.loads((root / 'inputs/cases.json').read_text())
            if c['split'] == 'holdout'
        ]
        for index, cell in enumerate(cases):
            order = ['H1', 'E1'] if index % 2 == 0 else ['E1', 'H1']
            before, after = (
                [
                    r
                    for r in turns
                    if r['identity']['phase'] == 'formal'
                    and r['identity']['version'] == v
                    and r['identity']['case'] == cell['id']
                    and r.get('evaluation')
                ]
                for v in order
            )
            assert len(before) == len(after) == len(cell['instructions'])
            assert max(r['verified_mono'] for r in before) < min(
                r['start_mono'] for r in after
            )
    write_json(
        root / 'metrics/audit.json',
        dict(
            status='passed',
            complete=complete,
            rendered_turns=len(rendered),
            one_model_request_per_completed_turn=all(
                len(r['requests']) == 1 for r in turns if r.get('evaluation')
            ),
            requests=summary['requests'],
            core_source_matches_frozen_version=True,
            formal_episode_order_matches_schedule=complete,
        ),
    )
    if not complete:
        print('Partial comparison saved')
        return
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )
    vs = ['H1', 'E1']
    colors = ['#607286', '#168b82']
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    axs[0].bar(
        vs, [summary['versions'][v]['successes'] for v in vs], color=colors
    )
    axs[0].set(
        ylim=(0, 26),
        ylabel='Correct whole tasks / 24',
        title='Quality, identical checks',
    )
    axs[1].bar(
        vs,
        [summary['versions'][v]['known_tokens'] / 32 / 1000 for v in vs],
        color=colors,
    )
    axs[1].set(
        ylabel='Known tokens / instruction (thousands)',
        title='Known usage only\n1 interrupted E1 request unknown',
    )
    local = [summary['versions'][v]['local_s'] / 32 for v in vs]
    model = [summary['versions'][v]['model_s'] / 32 for v in vs]
    axs[2].bar(
        vs, local, label='Local execution + verification', color='#9eb3c3'
    )
    axs[2].bar(
        vs, model, bottom=local, label='Model round trip', color='#35667d'
    )
    axs[2].set(
        ylabel='Seconds / instruction',
        title='Verified latency\nincluding interruption upper bound',
    )
    axs[2].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    for ax, values in [
        (axs[0], [summary['versions'][v]['successes'] for v in vs]),
        (
            axs[1],
            [summary['versions'][v]['known_tokens'] / 32 / 1000 for v in vs],
        ),
        (axs[2], [summary['versions'][v]['verified_s'] / 32 for v in vs]),
    ]:
        for i, value in enumerate(values):
            ax.text(i, value + 0.3, f'{value:.2f}', ha='center', va='bottom')
    fig.suptitle(
        'EmbodiedGen | Deterministic editing | GPU 4',
        fontsize=16,
        weight='bold',
    )
    fig.savefig(root / 'comparisons/results.png', dpi=170)
    plt.close(fig)
    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18
    )
    panels = []
    for v in ['H1', 'E1']:
        panels.append(
            (
                root
                / 'runs/formal'
                / v
                / 'D1_v2/turn_1/attempt_1/after_office.png',
                v + ': ground-lamp removal',
            )
        )
    sheet = Image.new('RGB', (1440, 520), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(panels):
        if path.exists():
            sheet.paste(
                Image.open(path).convert('RGB').resize((720, 480)),
                (720 * i, 40),
            )
        draw.text((720 * i + 10, 8), label, font=font, fill='#172331')
    sheet.save(root / 'comparisons/scene_comparison.jpg', quality=94)
    formal = [r for r in turns if r['identity']['phase'] == 'formal']
    for page in range((len(formal) + 11) // 12):
        sheet = Image.new('RGB', (1728, 960), '#f3f5f7')
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(formal[12 * page : 12 * (page + 1)]):
            i = r['identity']
            x, y = j % 4 * 432, j // 4 * 320
            path = Path(r['directory']) / 'after_office.png'
            if path.exists():
                sheet.paste(
                    Image.open(path).convert('RGB').resize((432, 288)),
                    (x, y + 30),
                )
            state = (
                'INTERRUPTED'
                if r['status'] == 'interrupted'
                else 'OK'
                if r.get('evaluation', {}).get('passed')
                else 'FAIL'
            )
            label = f"{i['version']} {i['case']} t{i['turn']} a{i['attempt']} {state}"
            draw.text((x + 5, y + 5), label, font=font, fill='#172331')
        sheet.save(
            root / f'comparisons/contact_{page + 1:02d}.jpg', quality=90
        )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != 'versions'}, indent=2
        )
    )


if __name__ == '__main__':
    main()
