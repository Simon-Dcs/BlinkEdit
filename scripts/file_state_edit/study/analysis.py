"""Descriptive statistics. Denominators stay fixed; subsets do not hide broken chains."""

import statistics
from collections import Counter
from pathlib import Path
from prepare import R, read, save
from ledger import main as ledger
import csv


def avg(seq):
    xs = [x for x in seq if x is not None]
    return sum(xs) / len(xs) if xs else None


def analyse():
    ledger()
    m = read(R / 'manifest.json')
    ts = list(csv.DictReader((R / 'metrics/edit2-turns.csv').open()))
    cs = {}
    for t in ts:
        for k in [
            'turn',
            'editor_calls',
            'reviewer_calls',
            'verifier_calls',
            'editor_known_tokens',
            'reviewer_known_tokens',
            'verifier_known_tokens',
        ]:
            t[k] = int(t[k])
        for k in [
            'IF',
            'CC',
            'CC_initial',
            'heldout_MAE',
            'elapsed_s',
            'editor_s',
            'reviewer_s',
            'verifier_s',
            'edit_s',
        ]:
            t[k] = float(t[k]) if t[k] else None
        for k in [
            'scored',
            'focus',
            'example',
            'completed',
            'accepted',
            'pending_review',
            'verified',
        ]:
            t[k] = (
                True if t[k] == 'True' else False if t[k] == 'False' else None
            )
        cs.setdefault((t['kind'], t['case'], t['arm']), []).append(t)
    zs = {}
    comparisons = {}
    cp = {}
    for kind, tag in [('edithero', 'edithero'), ('3dcodeedit', 'code')]:
        for g in ['simple', 'general']:
            keys = m[g][tag]
            for arm in m['arms']:
                cases = [cs.get((kind, k, arm), []) for k in keys]
                allturns = [t for c in cases for t in c]
                final = [c[-1] for c in cases if c]
                focus = [t for t in allturns if t['focus']]

                def covered(c):
                    if not c:
                        return False
                    z = c[-1]
                    return (
                        (
                            z['turn'] == m['chain_info'][z['case']]['n_turns']
                            and z['completed']
                        )
                        if kind == 'edithero'
                        else z['accepted'] is True
                        or z['turn'] >= m['max_revisions'] + 1
                    )

                name = f'{kind}/{g}/{arm}'
                z = dict(
                    assigned=len(keys),
                    started=sum(bool(c) for c in cases),
                    complete=sum(bool(covered(c)) for c in cases),
                    expected_turns=sum(
                        m['chain_info'][k]['n_turns'] for k in keys
                    )
                    if kind == 'edithero'
                    else len(keys),
                    attempts=len(allturns),
                    delivered=sum(bool(t['completed']) for t in allturns),
                    local=sum(
                        bool(t['completed']) and t['route'] == 'local'
                        for t in allturns
                    ),
                    focus_reached=sum(bool(t['completed']) for t in focus),
                    focus_local=sum(
                        bool(t['completed']) and t['route'] == 'local'
                        for t in focus
                    ),
                    scored=sum(bool(t['scored']) for t in allturns),
                    if_n=sum(t['IF'] is not None for t in allturns),
                    first_accepted=sum(
                        c[0]['accepted'] is True for c in cases if c
                    ),
                    repairs=sum(max(0, len(c) - 1) for c in cases),
                    judged=sum(t['verified'] is not None for t in final),
                    passed=sum(t['verified'] is True for t in final),
                    rejected=sum(t['verified'] is False for t in final),
                )
                for k in ['IF', 'CC', 'CC_initial', 'heldout_MAE']:
                    z[k] = avg(t[k] for t in allturns)
                    z['focal_' + k] = avg(t[k] for t in focus)
                for k in [
                    'editor_known_tokens',
                    'reviewer_known_tokens',
                    'verifier_known_tokens',
                    'editor_calls',
                    'reviewer_calls',
                    'verifier_calls',
                    'elapsed_s',
                    'editor_s',
                    'reviewer_s',
                    'verifier_s',
                ]:
                    z[k] = sum(t[k] for t in allturns)
                zs[name] = z
            # Fixed common set of fully finished cases for fair cost and quality comparisons.
            for a, b in [('F0', 'F0R'), ('F0R', 'F1R')]:
                common = [
                    k
                    for k in keys
                    if all(
                        covered(cs.get((kind, k, arm), [])) for arm in (a, b)
                    )
                ]
                d = {}
                for arm in (a, b):
                    t = [x for k in common for x in cs[kind, k, arm]]
                    f = [cs[kind, k, arm][-1] for k in common]
                    d[arm] = dict(
                        edit_tokens=sum(
                            x['editor_known_tokens']
                            + x['reviewer_known_tokens']
                            for x in t
                        ),
                        elapsed_s=sum(x['elapsed_s'] for x in t),
                        calls=sum(
                            x['editor_calls'] + x['reviewer_calls'] for x in t
                        ),
                        IF=avg(x['IF'] for x in t),
                        CC=avg(x['CC'] for x in t),
                        MAE=avg(x['heldout_MAE'] for x in t),
                        passed=sum(x['verified'] is True for x in f),
                        judged=sum(x['verified'] is not None for x in f),
                    )
                comparisons[f'{kind}/{g}/{a}-{b}'] = dict(
                    common=common, methods=d
                )
            for k in keys:
                cp[k] = {
                    arm: dict(
                        n=len(c),
                        local_turns=[
                            t['turn']
                            for t in c
                            if t['route'] == 'local' and t['completed']
                        ],
                        last=c[-1]['turn'] if c else None,
                        accepted=c[-1]['accepted'] if c else None,
                        verified=c[-1]['verified'] if c else None,
                        error=c[-1]['error'] if c else None,
                        first_accepted=c[0]['accepted'] if c else None,
                        IF=avg(t['IF'] for t in c),
                        CC=avg(t['CC'] for t in c),
                        MAE=avg(t['heldout_MAE'] for t in c),
                    )
                    for arm in m['arms']
                    for c in [cs.get((kind, k, arm), [])]
                }
    save(
        R / 'metrics/findings.json',
        dict(groups=zs, paired=comparisons, cases=cp),
    )
    return zs, comparisons, cp


if __name__ == '__main__':
    analyse()
