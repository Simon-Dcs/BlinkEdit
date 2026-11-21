"""Produce readable comparisons from retained evidence without new model calls."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from embodied_gen.harness.scene_edit.records import write_json


def csv_file(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def stats(values):
    return dict(
        mean=statistics.mean(values),
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root')
    a = p.parse_args()
    root = Path(
        a.root
        or Path('outputs/vista_evolution/latest_run.txt').read_text().strip()
    )
    records = [json.loads(p.read_text()) for p in root.rglob('record.json')]
    records = [r for r in records if 'model' in r and 'usage' in r]
    turns = [
        json.loads(p.read_text())
        for p in (root / 'episodes').glob('*/*/turn_*/attempt_*/result.json')
    ]
    turns.sort(key=lambda r: r['start_mono'])
    summaries = [
        json.loads(p.read_text())
        for p in (root / 'evaluations').glob('*/summary.json')
    ]
    summary = dict(
        requests=len(records),
        known_tokens=sum(
            (r.get('usage') or {}).get('input_tokens', 0)
            + (r.get('usage') or {}).get('output_tokens', 0)
            for r in records
        ),
        unknown_usage=sum(r.get('usage') is None for r in records),
        model_failures=sum(r['status'] != 'ok' for r in records),
        timed_turns=len(turns),
        selection={},
        final={},
        phases={},
        maximum_translation_m=0,
        maximum_rotation_deg=0,
    )
    flat = []
    for r in turns:
        identity = r['identity']
        e = r.get('evaluation', {})

        def use(key):
            return sum(
                (q.get('usage') or {}).get(key, 0) for q in r['requests']
            )

        flat.append(
            dict(
                **identity,
                instruction=r['instruction'],
                status=r['status'],
                passed=e.get('passed', False),
                critical=e.get('critical', False),
                reason=r.get('reason'),
                error=r.get('error'),
                verified_s=r['verified_s'],
                llm_rpc_s=r['llm_rpc_s'],
                non_llm_s=r['non_llm_s'],
                tokens=r['known_tokens'],
                input_tokens=use('input_tokens'),
                output_tokens=use('output_tokens'),
                cached_input_tokens=use('cached_input_tokens'),
                published=r['published'],
                output=r['output'],
                startup_s=r.get('startup_s'),
            )
        )
        for result in r.get('release', {}).values():
            summary['maximum_translation_m'] = max(
                summary['maximum_translation_m'], result['translation_m']
            )
            summary['maximum_rotation_deg'] = max(
                summary['maximum_rotation_deg'], result['rotation_deg']
            )
    csv_file(root / 'metrics/turns.csv', flat)
    csv_file(
        root / 'metrics/requests.csv',
        [
            {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            | (r.get('usage') or {})
            for r in records
        ],
    )
    episode_rows = []
    for s in summaries:
        for e in s['episodes']:
            episode_rows.append(
                dict(
                    evaluation=s['id'],
                    version=s['version'],
                    purpose=s['purpose'],
                    split=s['split'],
                    **e,
                )
            )
        if s['purpose'] not in ['formal', 'final']:
            continue
        edit = [
            e
            for e in s['episodes']
            if e['task'] not in ['B1', 'B2', 'B3', 'D3']
        ]
        boundary = [
            e for e in s['episodes'] if e['task'] in ['B1', 'B2', 'B3', 'D3']
        ]
        vals = [r for r in flat if r['evaluation_id'] == s['id']]
        # Quality uses the completed execution; cost includes every attempt.
        tokens = sum(r['tokens'] for r in vals)
        verified_s = sum(r['verified_s'] for r in vals)
        model_s = sum(r['llm_rpc_s'] for r in vals)
        compact = dict(
            successes=s['successes'],
            denominator=s['denominator'],
            edit_passed=sum(e['passed'] for e in edit),
            edit_tasks=len(edit),
            boundary_passed=sum(e['passed'] for e in boundary),
            boundary_tasks=len(boundary),
            tokens=tokens,
            verified_s=verified_s,
            mean_turn_s=verified_s / s['turns'],
            mean_turn_tokens=tokens / s['turns'],
            tokens_per_pass=tokens / s['successes']
            if s['successes']
            else None,
            attempt_time_distribution=stats([r['verified_s'] for r in vals]),
            episodes=s['episodes'],
            model_s=model_s,
            other_s=verified_s - model_s,
            execution_attempts=len(vals),
            completed_attempt_only_tokens=s['tokens'],
            completed_attempt_only_verified_s=s['verified_s'],
        )
        compact['session_wall_s'] = sum(
            e['session_wall_s'] for e in s['episodes']
        )
        compact['multi_turn_sessions'] = [
            {k: e[k] for k in ['id', 'turns', 'passed', 'session_wall_s']}
            for e in s['episodes']
            if e['turns'] > 1
        ]
        summary['selection' if s['purpose'] == 'formal' else 'final'][
            s['version']
        ] = compact
    csv_file(root / 'metrics/episodes.csv', episode_rows)
    # Count all attempts, including diagnostics and Meta calls, exactly once.
    for r in records:
        phase = r.get('stage', 'unknown')
        if r.get('phase') == 'meta' or phase.startswith('meta_'):
            phase = 'automatic_rewrite'
        elif phase == 'visibility_probe':
            phase = 'engineering'
        else:
            directory = r['directory']
            phase = next(
                (
                    name
                    for name in [
                        'engineering',
                        'development',
                        'formal',
                        'final',
                        'ablation',
                    ]
                    if '/' + name + '_' in directory
                ),
                'engineering' if '/setup/' in directory else 'other',
            )
        summary['phases'].setdefault(phase, dict(requests=0, tokens=0))
        summary['phases'][phase]['requests'] += 1
        summary['phases'][phase]['tokens'] += (r.get('usage') or {}).get(
            'input_tokens', 0
        ) + (r.get('usage') or {}).get('output_tokens', 0)
    decisions = [
        json.loads(p.read_text())
        for p in sorted((root / 'selections').glob('H*.json'))
    ]
    summary['decisions'] = decisions
    if (root / 'final_comparison.json').exists():
        final = json.loads((root / 'final_comparison.json').read_text())
        candidate = final['candidate_id']
        left = summary['final']['H0']
        right = summary['final'][candidate]
        recommendation = (
            candidate
            if final['selection_accepted']
            and not final['holdout_regressions']
            and right['successes'] >= left['successes']
            and final['candidate']['critical'] == 0
            else 'H0'
        )
        summary.update(
            final_candidate=candidate,
            recommended=recommendation,
            holdout_regressions=final['holdout_regressions'],
            final_token_change=right['tokens'] / left['tokens'] - 1,
            final_time_change=right['verified_s'] / left['verified_s'] - 1,
            tokens_per_pass_change=right['tokens_per_pass']
            / left['tokens_per_pass']
            - 1,
        )
        selection = json.loads((root / 'selected.json').read_text())
        chosen = (
            selection
            if recommendation != 'H0'
            else dict(id='H0', **json.loads((root / 'H0.json').read_text()))
        )
        write_json(root / 'recommended.json', chosen)
        comparisons = []
        initial_episodes = {e['id']: e for e in left['episodes']}
        revised_episodes = {e['id']: e for e in right['episodes']}
        for key in initial_episodes:
            paired_cost = {}
            for version in ['H0', candidate]:
                attempts = [
                    r
                    for r in flat
                    if r['evaluation_id'] == f'final_holdout_{version}'
                    and r['case'] == key
                ]
                paired_cost[version] = dict(
                    seconds=sum(r['verified_s'] for r in attempts),
                    tokens=sum(r['tokens'] for r in attempts),
                    attempts=len(attempts),
                )
            comparisons.append(
                dict(
                    case=key,
                    H0_passed=initial_episodes[key]['passed'],
                    candidate_passed=revised_episodes[key]['passed'],
                    H0_seconds=paired_cost['H0']['seconds'],
                    candidate_seconds=paired_cost[candidate]['seconds'],
                    H0_tokens=paired_cost['H0']['tokens'],
                    candidate_tokens=paired_cost[candidate]['tokens'],
                    H0_attempts=paired_cost['H0']['attempts'],
                    candidate_attempts=paired_cost[candidate]['attempts'],
                )
            )
        csv_file(root / 'metrics/final_pairs.csv', comparisons)
    write_json(root / 'metrics/summary.json', summary)
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    versions = sorted(summary['selection'])
    axes[0].bar(
        versions,
        [summary['selection'][v]['successes'] for v in versions],
        color=['#607286'] + ['#28928a'] * (len(versions) - 1),
    )
    axes[0].set(
        ylim=(0, 26),
        ylabel='Completed whole tasks / 24',
        title='Development layouts',
    )
    for i, v in enumerate(versions):
        axes[0].text(
            i,
            summary['selection'][v]['successes'] + 0.3,
            str(summary['selection'][v]['successes']),
            ha='center',
        )
    if summary['final']:
        vs = ['H0', summary['final_candidate']]
        axes[1].bar(
            vs,
            [summary['final'][v]['successes'] for v in vs],
            color=['#607286', '#28928a'],
        )
        axes[1].set(
            ylim=(0, 26),
            ylabel='Completed whole tasks / 24',
            title='New layouts, frozen versions',
        )
        for i, v in enumerate(vs):
            axes[1].text(
                i,
                summary['final'][v]['successes'] + 0.3,
                str(summary['final'][v]['successes']),
                ha='center',
            )
        axes[2].bar(
            vs,
            [summary['final'][v]['mean_turn_tokens'] / 1000 for v in vs],
            color=['#607286', '#28928a'],
        )
        axes[2].set(
            ylabel='Mean known tokens per instruction (thousands)',
            title='Cost includes failed instructions',
        )
        for i, v in enumerate(vs):
            axes[2].text(
                i,
                summary['final'][v]['mean_turn_tokens'] / 1000 + 0.1,
                f"{summary['final'][v]['mean_turn_tokens'] / 1000:.2f}k",
                ha='center',
            )
    fig.suptitle(
        'EmbodiedGen | Evidence-driven editing | GPU 4',
        fontsize=16,
        weight='bold',
    )
    fig.savefig(root / 'comparisons/results.png', dpi=170)
    plt.close(fig)
    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19
    )
    # Full evidence gallery: labels distinguish uncommitted failed candidates.
    meaningful = [
        r for r in turns if r['identity']['purpose'] in ['formal', 'final']
    ]
    for page in range((len(meaningful) + 11) // 12):
        sheet = Image.new('RGB', (1728, 960), '#f3f5f7')
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(meaningful[page * 12 : (page + 1) * 12]):
            i = r['identity']
            x, y = j % 4 * 432, j // 4 * 320
            d = (
                Path(next(q['directory'] for q in r['requests'])).parent.parent
                if r['requests']
                else None
            )
            if d and (d / 'after_office.png').exists():
                sheet.paste(
                    Image.open(d / 'after_office.png')
                    .convert('RGB')
                    .resize((432, 288)),
                    (x, y + 30),
                )
            label = f"{i['version']} {i['case']} t{i['turn']} " + (
                'OK' if r.get('evaluation', {}).get('passed') else 'FAIL'
            )
            draw.text((x + 6, y + 4), label, font=font, fill='#172331')
        sheet.save(
            root / f'comparisons/contact_{page + 1:02d}.jpg', quality=90
        )
    if summary.get('final_candidate'):
        v = summary['final_candidate']
        panels = []
        for version, filename, label in [
            ('H0', 'before_office.png', 'Input: two lamps in the office'),
            ('H0', 'after_office.png', 'Initial version: H0'),
            (v, 'after_office.png', 'Revised version: ' + v),
        ]:
            d = (
                root
                / 'episodes'
                / f'final_holdout_{version}'
                / 'D1_v2'
                / 'turn_1'
                / 'attempt_1'
            )
            if (d / filename).exists():
                panels.append((d / filename, label))
        if len(panels) == 3:
            sheet = Image.new('RGB', (3 * 720, 520), 'white')
            draw = ImageDraw.Draw(sheet)
            for j, (p, label) in enumerate(panels):
                sheet.paste(
                    Image.open(p).convert('RGB').resize((720, 480)),
                    (j * 720, 40),
                )
                draw.text((j * 720 + 10, 9), label, font=font, fill='#172331')
            sheet.save(root / 'comparisons/grounding.jpg', quality=94)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ['selection', 'final']
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
