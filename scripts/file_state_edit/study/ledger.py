"""Join stage roles, frozen strata and completeness to paired exposure. No inference calls."""

import csv, json, sys
from pathlib import Path
from prepare import R, read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize import summarize, save_csv


def main():
    m = read(R / 'manifest.json')
    summarize(R, 'formal')
    focus = m['focus']
    rows = []
    requests = {}
    rp = R / 'metrics/formal-requests.csv'
    if rp.exists():
        with rp.open() as f:
            for r in csv.DictReader(f):
                q = (r['kind'], r['case'], r['arm'], int(r['turn']))
                requests.setdefault(q, []).append(r)
    for kind, tag in [('edithero', 'edithero'), ('3dcodeedit', 'code')]:
        for k in m['formal'][tag]:
            group = 'simple' if k in m['simple'][tag] else 'general'
            for arm in m['arms']:
                d = R / kind / 'formal' / k / arm
                # Deferred outputs have an independently measured edit/render stage.
                # Do not charge a valid reviewer twice if a transport attempt was retried.
                deferred = {}
                if kind == '3dcodeedit' and (d / 'worker.log').is_file():
                    for line in (d / 'worker.log').read_text().splitlines():
                        if line.startswith('{"kind"'):
                            stage = json.loads(line)
                            if stage.get('review_pending'):
                                deferred[stage['turn']] = stage['elapsed_s']
                for p in sorted(d.glob('turn*/outcome.json')):
                    v = read(p)
                    turn = v['turn']
                    score = (
                        read(p.parent / 'score.json')
                        if (p.parent / 'score.json').is_file()
                        else {}
                    )
                    q = requests.get((kind, k, arm, turn), [])
                    z = dict(
                        kind=kind,
                        stratum=group,
                        case=k,
                        arm=arm,
                        turn=turn,
                        focus=(kind == 'edithero' and focus.get(k) == turn),
                        example=k in m['known_examples'],
                        route=v.get('route'),
                        completed=v.get('completed'),
                        accepted=v.get('accept'),
                        pending_review=v.get('review_pending', False),
                        error=v.get('error'),
                        state_cache=v.get('cached'),
                        elapsed_s=v['elapsed_s'],
                        raw_active_elapsed_s=v['elapsed_s'],
                        timing_basis='continuous',
                        edit_s=v.get('edit_s'),
                        cache_setup_s=v.get('state_setup_s'),
                        scored=(p.parent / 'score.json').is_file(),
                        verified=score.get('all_pass'),
                        IF=score.get('IF'),
                        CC=score.get('CC_prev'),
                        CC_initial=score.get('CC_initial'),
                    )
                    mae = [
                        a['edit_MAE']
                        for a in score.get('views', [])
                        if a['view'] and a['edit_MAE'] is not None
                    ]
                    z['heldout_MAE'] = sum(mae) / len(mae) if mae else None
                    for role in ['editor', 'reviewer', 'verifier']:
                        items = [r for r in q if r['role'] == role]
                        z[role + '_calls'] = len(items)
                        z[role + '_known_tokens'] = sum(
                            int(r['known_tokens'] or 0) for r in items
                        )
                        z[role + '_unknown'] = sum(
                            not r['input'] for r in items
                        )
                        z[role + '_s'] = sum(float(r['time_s']) for r in items)
                    if turn in deferred and not z['pending_review'] and z['reviewer_calls'] == 1 and not z['reviewer_unknown']:
                        z['elapsed_s'] = deferred[turn] + z['reviewer_s']
                        z['timing_basis'] = 'offline_edit_render_plus_valid_reviewer'
                    rows.append(z)
    save_csv(R / 'metrics/edit2-turns.csv', rows)
    summary = []
    for kind, tag in [('edithero', 'edithero'), ('3dcodeedit', 'code')]:
        for stratum in ['simple', 'general']:
            for arm in m['arms']:
                assigned = m[stratum][tag]
                items = [
                    z
                    for z in rows
                    if z['kind'] == kind
                    and z['stratum'] == stratum
                    and z['arm'] == arm
                ]
                keys = set(z['case'] for z in items)
                finals = [
                    max(
                        (z for z in items if z['case'] == k),
                        key=lambda z: z['turn'],
                    )
                    for k in keys
                ]
                summary.append(
                    dict(
                        kind=kind,
                        stratum=stratum,
                        arm=arm,
                        assigned=len(assigned),
                        started=len(keys),
                        fully_delivered=sum(
                            (
                                r['turn']
                                == m['chain_info'][r['case']]['n_turns']
                                and r['completed']
                            )
                            if kind == 'edithero'
                            else (
                                r['accepted']
                                or r['turn'] == 1 + m['max_revisions']
                            )
                            is True
                            for r in finals
                        ),
                        executed_edits=sum(
                            bool(z['completed']) for z in items
                        ),
                        local_edits=sum(
                            z['completed'] and z['route'] == 'local'
                            for z in items
                        ),
                        graded=sum(z['verified'] is not None for z in finals),
                        checks_pass=sum(z['verified'] is True for z in finals),
                        pending_reviews=sum(
                            z['pending_review'] for z in items
                        ),
                        focus_delivered=sum(
                            z['focus'] and z['completed'] for z in items
                        ),
                        **{
                            k: sum(z[k] for z in items)
                            for k in [
                                'elapsed_s',
                                'editor_calls',
                                'reviewer_calls',
                                'verifier_calls',
                                'editor_known_tokens',
                                'reviewer_known_tokens',
                                'verifier_known_tokens',
                                'editor_unknown',
                                'reviewer_unknown',
                                'verifier_unknown',
                            ]
                        },
                    )
                )
    save_csv(R / 'metrics/edit2-groups.csv', summary)


if __name__ == '__main__':
    main()
