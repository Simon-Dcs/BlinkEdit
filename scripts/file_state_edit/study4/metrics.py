"""Count real successful and failed stages; keep environment interruptions separate from benchmark results."""

import csv, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
from design import R, read, write


def export(name, rows):
    if not rows:
        return
    with (R / 'metrics' / name).open('w') as f:
        w = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for v in rows for k in v))
        )
        w.writeheader()
        w.writerows(rows)


def mean(values):
    return statistics.mean(values) if values else None


def collect(final=False):
    m = read(R / 'manifest.json')
    recs = []
    turns = []
    cases = []
    trials = []
    pairs = []
    issues = []

    def scan(out, **meta):
        rows = []
        for p in sorted(out.rglob('record.json')):
            v = read(p)
            u = v.get('usage') or {}
            cmd = read(p.with_name('command.json'))
            gate = p.parent.parent / 'admission.json'
            q = read(gate).get('queue_s', 0) if gate.is_file() else 0
            z = dict(
                **meta,
                stage=v['stage'],
                component=p.relative_to(out).parts[0],
                status=v['status'],
                known=bool(u),
                queue=q,
                input=u.get('input_tokens', 0),
                output=u.get('output_tokens', 0),
                cached=u.get('cached_input_tokens', 0),
                tokens=u.get('input_tokens', 0) + u.get('output_tokens', 0),
                images=sum(x in ('--image', '-i') for x in cmd),
                latency=v.get('latency_s') or 0,
                path=str(p.relative_to(R)),
            )
            rows.append(z)
            recs.append(z)
        d = {}
        for stage in ('editor', 'reviewer', 'verifier'):
            xs = [z for z in rows if z['stage'] == stage]
            d[stage + '_calls'] = len(xs)
            for unit in [
                'tokens',
                'input',
                'output',
                'images',
                'latency',
                'queue',
            ]:
                d[stage + '_' + unit] = sum(z[unit] for z in xs)
        return d

    ref = {
        v['case_id']: v
        for v in map(
            json.loads, (Path(m['roots']['code']) / 'data/tasks.jsonl').open()
        )
    }
    for k in m['formal']['code']:
        pop = (
            'fresh'
            if k in m['fresh']['code']
            else 'simple'
            if k in m['simple']['code']
            else 'general'
        )
        for arm in m['arms']:
            ds = sorted(
                (R / '3dcodeedit/formal' / k / arm).glob('turn*/outcome.json')
            )
            ts = []
            for p in ds:
                v = read(p)
                out = p.parent
                s = (
                    read(out / 'score.json')
                    if (out / 'score.json').exists()
                    else {}
                )
                b = (
                    read(out / 'binding.json')
                    if (out / 'binding.json').exists()
                    else {}
                )
                a = dict(
                    kind='code',
                    case=k,
                    arm=arm,
                    population=pop,
                    turn=v['turn'],
                )
                t = dict(
                    **a,
                    route=v.get('route'),
                    done=v.get('completed'),
                    accept=v.get('accept'),
                    secs=v['elapsed_s'],
                    error=v.get('error'),
                    judged=bool(s),
                    passed=s.get('all_pass'),
                    nitems=len(s.get('items', [])),
                    trace_s=b.get('seconds', 0),
                    trace_why=b.get('reason'),
                    objects=b.get('objects'),
                    mapped=b.get('mapped'),
                    grounded=bool(b.get('spans')),
                    patch=(out / 'patch.json').is_file(),
                    **scan(out, **a),
                )
                ts.append(t)
                turns.append(t)
            x = ts[-1] if ts else {}
            terminal = bool(
                ts and (x['accept'] or x['turn'] == 1 + m['max_revisions'])
            )
            if not (
                terminal
                and x['judged']
                and (not x['done'] or x['nitems'] == len(ref[k]['checklist']))
            ):
                issues.append('pending ' + k + '/' + arm)
            c = dict(
                case=k,
                arm=arm,
                population=pop,
                turns=len(ts),
                started=bool(ts),
                terminal=terminal,
                judged=x.get('judged', False),
                passed=x.get('passed'),
                first_accept=bool(ts and ts[0]['accept']),
                repairs=max(0, len(ts) - 1),
                trace_attempts=sum(t['trace_why'] is not None for t in ts),
                grounded=sum(t['grounded'] for t in ts),
                patch_tries=sum(t['patch'] for t in ts),
                merged=sum(t['route'] == 'model_patch' for t in ts),
                rendered=sum(bool(t['done']) for t in ts),
                first_editor=ts[0]['editor_tokens'] if ts else 0,
            )
            for u in [
                'secs',
                'trace_s',
                *[
                    s + '_' + j
                    for s in ['editor', 'reviewer', 'verifier']
                    for j in [
                        'calls',
                        'input',
                        'output',
                        'tokens',
                        'images',
                        'latency',
                        'queue',
                    ]
                ],
            ]:
                c[u] = sum(t[u] for t in ts)
            c['service_s'] = (
                c['secs'] - c['editor_queue'] - c['reviewer_queue']
            )
            c['delivery'] = c['editor_tokens'] + c['reviewer_tokens']
            cases.append(c)
    for v in m['paired_trials']:
        pair = []
        for method in v['order']:
            arm = f"p{v['i']}r{v['rep']}-{method}"
            out = (
                R / 'edithero/formal' / v['case'] / arm / f"turn{v['turn']:02}"
            )
            if not (out / 'pair-score.json').is_file():
                issues.append('pending ' + arm)
                continue
            z = read(out / 'outcome.json')
            s = read(out / 'pair-score.json')
            a = dict(
                kind='paired',
                case=v['case'],
                arm=method,
                population='states',
                turn=v['turn'],
                rep=v['rep'],
                state=v['i'],
            )
            x = dict(
                **a,
                first=v['order'][0] == method,
                route=z.get('route'),
                seconds=z['elapsed_s'],
                done=z['completed'],
                error=z.get('error'),
                available=s['available'],
                exact=s['exact'],
                heldout_mae=s.get('heldout_mae'),
                **scan(out, **a),
            )
            if (
                not z['completed']
                and z.get('error')
                and not (out / 'actions.json').is_file()
            ):
                issues.append('inspect ' + arm)
            x['wall_s'] = x['seconds']
            x['seconds'] -= x['editor_queue']
            trials.append(x)
            pair.append(x)
        if len(pair) == 2:
            f = next(x for x in pair if x['arm'] == 'full')
            l = next(x for x in pair if x['arm'] == 'select')
            pairs.append(
                dict(
                    state=v['i'],
                    rep=v['rep'],
                    case=v['case'],
                    turn=v['turn'],
                    first=v['order'][0],
                    sec_full=f['seconds'],
                    sec_select=l['seconds'],
                    delta_s=l['seconds'] - f['seconds'],
                    full_correct=f['exact'],
                    select_correct=l['exact'],
                    full_tokens=f['editor_tokens'],
                    select_tokens=l['editor_tokens'],
                )
            )
    groups = {}
    for pop in ['simple', 'general', 'main', 'fresh', 'all']:
        groups[pop] = {}
        for arm in m['arms']:
            xs = [
                c
                for c in cases
                if c['arm'] == arm
                and (
                    pop == 'all'
                    or c['population'] == pop
                    or (pop == 'main' and c['population'] != 'fresh')
                )
            ]
            groups[pop][arm] = dict(
                assigned=len(xs),
                **{
                    k: sum(c[k] or 0 for c in xs)
                    for k in xs[0]
                    if k not in ['case', 'arm', 'population']
                },
            )
    budget = read(R / 'metrics/formal-budget.json')
    roles = defaultdict(lambda: dict(calls=0, tokens=0))
    missing = []
    latest = budget['start']
    tools = 0
    for d in map(Path, budget['invocations']):
        ps = list(d.rglob('record.json'))
        if len(ps) != 1:
            missing.append(str(d.relative_to(R)))
        for p in ps:
            latest = max(latest, p.stat().st_mtime)
            v = read(p)
            u = v.get('usage') or {}
            if not u:
                missing.append(str(p.relative_to(R)))
            z = roles[v['stage']]
            z['calls'] += 1
            z['tokens'] += u.get('input_tokens', 0) + u.get('output_tokens', 0)
            tools += len(v.get('tool_items', []))
    endpoints = list(
        (R / '3dcodeedit/formal').glob('*/*/turn*/score.json')
    ) + list((R / 'edithero/formal').rglob('pair-score.json'))
    summary = dict(
        start=budget['start'],
        elapsed=max([latest] + [p.stat().st_mtime for p in endpoints])
        - budget['start'],
        calls=len(budget['invocations']),
        tool_actions=tools,
        known_tokens=sum(x['tokens'] for x in roles.values()),
        roles=dict(roles),
        missing=missing,
    )
    export('requests.csv', recs)
    export('program-turns.csv', turns)
    export('program-cases.csv', cases)
    export('state-trials.csv', trials)
    export('paired.csv', pairs)
    write(R / 'metrics/groups.json', groups)
    write(R / 'metrics/resource.json', summary)
    write(
        R / 'metrics/coverage.json',
        dict(
            pending=issues,
            judged=sum(c['judged'] for c in cases),
            paired=len(pairs),
        ),
    )
    if final and issues:
        raise RuntimeError(str(issues[:12]))
    return groups, cases, turns, trials, pairs, summary


if __name__ == '__main__':
    g, c, t, x, p, b = collect()
    print(
        'judged',
        sum(i['judged'] for i in c),
        'pairs',
        len(p),
        'new attempts',
        b['calls'],
        'tokens',
        b['known_tokens'],
    )
