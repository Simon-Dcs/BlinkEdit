"""Four-way ledger; persist provenance on every turn. This only reads experiment records."""

import collections
import csv
import statistics

from setup_run import PREV, ROOT, read, save

R = ROOT


def csvout(name, rows):
    if not rows:
        return
    with (R / 'metrics' / name).open('w') as f:
        w = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def numeric(rec, role, k):
    return sum(x[k] for x in rec if x['role'] == role)


def collect():
    m = read(R / 'manifest.json')
    prev_clock = {
        (z['kind'], z['case'], z['arm'], int(z['turn'])): float(z['elapsed_s'])
        for z in csv.DictReader((PREV / 'metrics/edit2-turns.csv').open())
    }
    rows = []
    receipts = []
    cases = []
    groups = {}
    paired = {}
    for kind, tag in [('edithero', 'edithero'), ('3dcodeedit', 'code')]:
        for k in m['formal'][tag]:
            pop = (
                'fresh'
                if k in m['fresh'][tag]
                else 'simple'
                if k in m['simple'][tag]
                else 'general'
            )
            for arm in m['arms']:
                historic = arm not in m['run_arms'][k]
                src = PREV if historic else R
                alias = m['prior_arms'][arm] if historic else arm
                case = src / kind / 'formal' / k / alias
                ts = []
                for p in sorted(case.glob('turn*/outcome.json')):
                    v = read(p)
                    d = p.parent
                    s = (
                        read(d / 'score.json')
                        if (d / 'score.json').exists()
                        else {}
                    )
                    rec = []
                    for log in sorted(d.rglob('record.json')):
                        z = read(log)
                        u = z.get('usage') or {}
                        cmd = read(log.parent / 'command.json')
                        rr = dict(
                            kind=kind,
                            case=k,
                            arm=arm,
                            stratum=pop,
                            turn=v['turn'],
                            role=z['stage'],
                            source=src.name,
                            known=bool(u),
                            ok=z.get('status') == 'ok',
                            input=u.get('input_tokens', 0),
                            output=u.get('output_tokens', 0),
                            cached=u.get('cached_input_tokens', 0),
                            tokens=u.get('input_tokens', 0)
                            + u.get('output_tokens', 0),
                            images=sum(a in ('--image', '-i') for a in cmd),
                            tool_actions=len(z.get('tool_items', [])),
                            latency=z['latency_s'],
                            component=log.relative_to(d).parts[0],
                            path=str(log.relative_to(R.parent)),
                        )
                        rec.append(rr)
                        receipts.append(rr)
                    t = dict(
                        kind=kind,
                        case=k,
                        arm=arm,
                        stratum=pop,
                        turn=v['turn'],
                        focus=m['focus'].get(k) == v['turn']
                        if tag == 'edithero'
                        else False,
                        route=v.get('route'),
                        source=src.name,
                        path=str(d.relative_to(R.parent)),
                        done=v['completed'],
                        accept=v.get('accept'),
                        error=v.get('error'),
                        service_s=prev_clock[kind, k, alias, v['turn']]
                        if historic
                        else v['elapsed_s'],
                        raw_s=v['elapsed_s'],
                        scored=bool(s),
                        passed=s.get('all_pass'),
                        IF=s.get('IF'),
                        CC=s.get('CC_prev'),
                        CC_initial=s.get('CC_initial'),
                        MAE=mean(
                            z['edit_MAE']
                            for z in s.get('views', [])
                            if z['view']
                        ),
                        grade_s=s.get('score_s', 0),
                        items=len(s.get('items', [])),
                        patch_attempt=(d / 'patch.json').exists(),
                        selector_attempt=(d / 'locator').is_dir(),
                        local_measure=(d / 'grounding.json').is_file()
                        and v.get('route') == 'measured_remove',
                    )
                    for role in ['editor', 'reviewer', 'verifier']:
                        t[role + '_calls'] = sum(
                            z['role'] == role for z in rec
                        )
                        for name in [
                            'tokens',
                            'images',
                            'latency',
                            'tool_actions',
                        ]:
                            t[role + '_' + name] = numeric(rec, role, name)
                    ts.append(t)
                    rows.append(t)
                last = ts[-1] if ts else {}
                terminal = bool(
                    ts
                    and (
                        (
                            ts[-1]['done']
                            and ts[-1]['turn'] == m['chain_info'][k]['n_turns']
                        )
                        if tag == 'edithero'
                        else (
                            ts[-1]['accept']
                            or ts[-1]['turn'] == m['max_revisions'] + 1
                        )
                    )
                )
                cases.append(
                    dict(
                        kind=kind,
                        case=k,
                        arm=arm,
                        stratum=pop,
                        source=src.name,
                        turns=len(ts),
                        delivered=sum(t['done'] for t in ts),
                        terminal=terminal,
                        first_accept=bool(ts and ts[0]['accept']),
                        passed=last.get('passed'),
                        error=last.get('error'),
                        repairs=max(0, len(ts) - 1) if tag == 'code' else 0,
                    )
                )
            # per-key inventory is complete before aggregation

    def agg(ts, cs):
        return dict(
            assigned=len(cs),
            complete=sum(c['terminal'] for c in cs),
            attempts=len(ts),
            done=sum(t['done'] for t in ts),
            scored=sum(t['scored'] for t in ts),
            passed=sum(c['passed'] is True for c in cs),
            judged=sum(c['passed'] is not None for c in cs),
            first=sum(c['first_accept'] for c in cs),
            repairs=sum(c['repairs'] for c in cs),
            items=sum(x['items'] for x in ts),
            **{
                k: sum(t[k] for t in ts)
                for k in [
                    'service_s',
                    'grade_s',
                    'editor_calls',
                    'reviewer_calls',
                    'verifier_calls',
                    'editor_tokens',
                    'reviewer_tokens',
                    'verifier_tokens',
                    'editor_images',
                    'reviewer_images',
                    'verifier_images',
                    'editor_tool_actions',
                    'editor_latency',
                    'reviewer_latency',
                    'verifier_latency',
                ]
            },
            **{
                k: mean(t[k] for t in ts)
                for k in ['IF', 'CC', 'CC_initial', 'MAE']
            },
            n_IF=sum(t['IF'] is not None for t in ts),
            routes=dict(collections.Counter(str(t['route']) for t in ts)),
            patch_attempts=sum(t['patch_attempt'] for t in ts),
            selector_attempts=sum(t['selector_attempt'] for t in ts),
        )

    for kind in ['edithero', '3dcodeedit']:
        for pop in ['simple', 'general', 'fresh', 'main']:
            ts = [
                t
                for t in rows
                if t['kind'] == kind
                and (
                    pop == t['stratum']
                    or (pop == 'main' and t['stratum'] != 'fresh')
                )
            ]
            cs = [
                c
                for c in cases
                if c['kind'] == kind
                and (
                    pop == c['stratum']
                    or (pop == 'main' and c['stratum'] != 'fresh')
                )
            ]
            key = f'{kind}/{pop}'
            groups[key] = {
                a: agg(
                    [t for t in ts if t['arm'] == a],
                    [c for c in cs if c['arm'] == a],
                )
                for a in m['arms']
            }
            for name, arms in [('all', m['arms']), ('F1-F2', ['F1', 'F2'])]:
                common = sorted(
                    set.intersection(
                        *[
                            {
                                c['case']
                                for c in cs
                                if c['terminal'] and c['arm'] == a
                            }
                            for a in arms
                        ]
                    )
                )
                paired[key + '/' + name] = dict(
                    keys=common,
                    methods={
                        a: agg(
                            [
                                t
                                for t in ts
                                if t['case'] in common and t['arm'] == a
                            ],
                            [
                                c
                                for c in cs
                                if c['case'] in common and c['arm'] == a
                            ],
                        )
                        for a in arms
                    },
                )
    csvout('four-turns.csv', rows)
    csvout('four-cases.csv', cases)
    csvout('four-requests.csv', receipts)
    save(R / 'metrics/groups.json', groups)
    save(R / 'metrics/common.json', paired)
    return rows, cases, receipts, groups, paired


if __name__ == '__main__':
    rows, cases, receipts, g, p = collect()
    print(
        'this run',
        sum(x['source'] == R.name for x in receipts),
        'attributed receipts; main',
        {
            k: [(a, v['complete'], v['scored']) for a, v in vs.items()]
            for k, vs in g.items()
            if k.endswith('/main')
        },
    )
