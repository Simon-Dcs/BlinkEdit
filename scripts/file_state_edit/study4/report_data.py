"""Produce the paired contrasts and stage accounting used in edit4.md (local only)."""

from collections import Counter
from statistics import median
from design import R, read, write
from metrics import collect


def stats():
    _, cases, turns, trial, pairs, res = collect()
    m = read(R / 'manifest.json')
    out = {
        'by_pair': {},
        'paired': {},
        'mechanism': {},
        'contrasts': {},
        'groups': {},
        'cost': {},
    }
    for i in range(8):
        p = [v for v in pairs if v['state'] == i]
        ts = [x for x in trial if x['state'] == i]
        if not p:
            continue
        out['by_pair'][i] = dict(
            n=len(p),
            delta=median(v['delta_s'] for v in p),
            **{
                how: dict(
                    exact=sum(v['exact'] for v in ts if v['arm'] == how),
                    sec_med=median(
                        v['seconds'] for v in ts if v['arm'] == how
                    ),
                    targets_present=sum(
                        v['available'] for v in ts if v['arm'] == how
                    ),
                    routes=dict(
                        Counter(v['route'] for v in ts if v['arm'] == how)
                    ),
                )
                for how in ('full', 'select')
            },
        )
    for how in ('full', 'select'):
        ts = [x for x in trial if x['arm'] == how]
        if not ts:
            continue
        out['paired'][how] = {
            k: sum(t[k] for t in ts)
            for k in [
                'editor_tokens',
                'editor_input',
                'editor_output',
                'editor_images',
                'editor_calls',
                'editor_latency',
                'editor_queue',
                'wall_s',
                'seconds',
                'exact',
                'available',
            ]
        }
        out['paired'][how]['heldout_mean'] = sum(
            t['heldout_mae'] for t in ts if t['heldout_mae'] is not None
        ) / max(1, sum(t['heldout_mae'] is not None for t in ts))
        out['paired'][how]['routes'] = dict(Counter(t['route'] for t in ts))
    if pairs:
        out['paired']['contrast'] = dict(
            n=len(pairs),
            median_delta=median(v['delta_s'] for v in pairs),
            faster=sum(v['delta_s'] < 0 for v in pairs),
            only_select=sum(
                v['select_correct'] and not v['full_correct'] for v in pairs
            ),
            only_full=sum(
                v['full_correct'] and not v['select_correct'] for v in pairs
            ),
            both=sum(v['full_correct'] and v['select_correct'] for v in pairs),
            neither=sum(
                not v['full_correct'] and not v['select_correct']
                for v in pairs
            ),
        )
        out['paired']['by_order'] = {
            how: dict(
                n=len(vs), median_delta_s=median(z['delta_s'] for z in vs)
            )
            for how in ('full', 'select')
            if (vs := [z for z in pairs if z['first'] == how])
        }
    for pop in ('simple', 'general', 'main', 'fresh', 'all'):
        xs = [
            c
            for c in cases
            if pop == 'all'
            or c['population'] == pop
            or pop == 'main'
            and c['population'] != 'fresh'
        ]
        for a in m['arms']:
            cs = [v for v in xs if v['arm'] == a]
            out['groups'][pop + '/' + a] = dict(
                n=len(cs),
                **{
                    k: sum(v[k] or 0 for v in cs)
                    for k in [
                        'first_accept',
                        'repairs',
                        'passed',
                        'judged',
                        'first_editor',
                        'editor_tokens',
                        'reviewer_tokens',
                        'verifier_tokens',
                        'delivery',
                        'secs',
                        'service_s',
                        'editor_queue',
                        'reviewer_queue',
                        'trace_s',
                        'editor_calls',
                        'reviewer_calls',
                        'verifier_calls',
                        'reviewer_images',
                    ]
                },
            )
        done = {
            a: {v['case'] for v in xs if v['arm'] == a and v['passed']}
            for a in m['arms']
        }
        for a in ('F1', 'F2'):
            subset = done[a] & done['F2T']
            out['contrasts'][pop + '/' + a] = dict(
                both=len(subset),
                only_base=sorted(done[a] - done['F2T']),
                only_trace=sorted(done['F2T'] - done[a]),
                cost={
                    arm: [
                        sum(
                            v['delivery']
                            for v in xs
                            if v['case'] in subset and v['arm'] == arm
                        ),
                        sum(
                            v['service_s']
                            for v in xs
                            if v['case'] in subset and v['arm'] == arm
                        ),
                    ]
                    for arm in (a, 'F2T')
                },
            )
    for a in ('F2', 'F2T'):
        ts = [v for v in turns if v['arm'] == a]
        fs = [v for v in ts if v['turn'] == 1]

        def one(vs):
            return dict(
                routes=dict(Counter(v['route'] for v in vs)),
                binding=dict(Counter(v['trace_why'] for v in vs)),
                patch_tried=sum(v['patch'] for v in vs),
                patch_used=sum(v['route'] == 'model_patch' for v in vs),
                trace_s=sum(v['trace_s'] for v in vs),
                patch_accepted=sum(
                    bool(v.get('accept'))
                    for v in vs
                    if v['route'] == 'model_patch'
                ),
            )

        out['mechanism'][a] = dict(first=one(fs), all=one(ts))
    paths = (
        [(p, read(p)) for p in (R / 'diagnostics').rglob('record.json')]
        if (R / 'diagnostics').exists()
        else []
    )
    out['cost'] = dict(
        **res,
        archive=dict(
            calls=len(paths),
            timedout=sum(
                'TimeoutExpired' in v.get('error', '') for _, v in paths
            ),
            known_tokens=sum(
                sum(
                    (v.get('usage') or {}).get(k, 0)
                    for k in ['input_tokens', 'output_tokens']
                )
                for _, v in paths
            ),
            measured_model_s=sum(v.get('latency_s') or 0 for _, v in paths),
        ),
    )
    write(R / 'metrics/contrasts.json', out)
    return out


if __name__ == '__main__':
    stats()
