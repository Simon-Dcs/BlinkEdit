"""Aggregate mechanism-specific evidence; do not assign technical retries to new edits."""

from pathlib import Path

from collect import R, collect, read, save


def analyse():
    t, c, q, g, p = collect()
    e = {'mechanisms': {}, 'examples': {}, 'failures': []}
    for kind in ['edithero', '3dcodeedit']:
        for pop in ['simple', 'general', 'fresh']:
            ts = [
                x
                for x in t
                if x['arm'] == 'F2'
                and x['kind'] == kind
                and x['stratum'] == pop
            ]
            groups = {}
            for route in sorted({str(x['route']) for x in ts}):
                xs = [x for x in ts if str(x['route']) == route]
                ifs = [x['IF'] for x in xs if x['IF'] is not None]
                groups[route] = dict(
                    n=len(xs),
                    completed=sum(x['done'] for x in xs),
                    valid_IF=len(ifs),
                    IF_one=sum(i >= 0.999999 for i in ifs),
                    IF_zero=sum(i == 0 for i in ifs),
                    mean_IF=sum(ifs) / len(ifs) if ifs else None,
                    review_pass=sum(x['accept'] is True for x in xs),
                    e_tokens=sum(x['editor_tokens'] for x in xs),
                    e_calls=sum(x['editor_calls'] for x in xs),
                )
            e['mechanisms'][kind + '/' + pop] = groups
    m = read(R / 'manifest.json')
    for k in [
        'R/3ce5c791_s0',
        'R/f76addb8_s0',
        'openscad_tool-shelf-for-bench_L4_Q0',
        *m['fresh']['code'],
        *m['fresh']['edithero'],
    ]:
        e['examples'][k] = {}
        for a in m['arms']:
            xs = [x for x in t if x['case'] == k and x['arm'] == a]
            e['examples'][k][a] = dict(
                turns=len(xs),
                route=[x['route'] for x in xs],
                IF=[x['IF'] for x in xs],
                CC=[x['CC'] for x in xs],
                passed=xs[-1]['passed'] if xs else None,
                edit=sum(x['editor_tokens'] for x in xs),
                feedback=sum(x['reviewer_tokens'] for x in xs),
                service_s=sum(x['service_s'] for x in xs),
            )
    e['failures'] = [
        {
            k: c0[k]
            for k in [
                'kind',
                'case',
                'arm',
                'stratum',
                'turns',
                'error',
                'passed',
            ]
        }
        for c0 in c
        if c0['error'] or c0['passed'] is False
    ]
    e['budget'] = {}
    budget = read(R / 'metrics/formal-budget.json')
    rec = [x for x in q if x['source'] == R.name]
    # Each reservation owns exactly one CLI record, sometimes nested under request_00.
    missing = []
    last = budget['start']
    all_recs = []
    for d in map(Path, budget['invocations']):
        rr = list(d.rglob('record.json'))
        if not rr:
            missing.append(str(d.relative_to(R)))
            continue
        for z in rr:
            v = read(z)
            last = max(last, z.stat().st_mtime)
            all_recs.append(v)
            if not v.get('usage'):
                missing.append(str(z.relative_to(R)))
    roles = {}
    for s in ['editor', 'reviewer', 'verifier']:
        xs = [v for v in rec if v['role'] == s]
        roles[s] = dict(
            calls=len(xs),
            tokens=sum(v['tokens'] for v in xs),
            images=sum(v['images'] for v in xs),
            latency=sum(v['latency'] for v in xs),
        )
    e['budget'] = dict(
        reservations=len(budget['invocations']),
        tools=sum(len(v.get('tool_items', [])) for v in all_recs),
        known_tokens=sum(
            (v.get('usage') or {}).get('input_tokens', 0)
            + (v.get('usage') or {}).get('output_tokens', 0)
            for v in all_recs
        ),
        missing=missing,
        roles=roles,
        start=budget['start'],
        last_receipt=last,
        last_score=max(
            [last]
            + [
                z.stat().st_mtime
                for kind in ['3dcodeedit', 'edithero']
                for z in (R / kind).rglob('score.json')
            ]
        ),
    )
    save(R / 'metrics/evidence.json', e)
    return t, c, q, g, p, e


if __name__ == '__main__':
    analyse()
