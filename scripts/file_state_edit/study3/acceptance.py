"""Validate the terminal report population, scored outputs and role costs, including interruptions."""

import json
from pathlib import Path

from evidence import analyse
from setup_run import ROOT as R
from setup_run import read


def check():
    ts, cs, qs, g, p, e = analyse()
    m = read(R / 'manifest.json')
    errors = []

    def must(cond, msg):
        if not cond:
            errors.append(msg)

    for c in cs:
        kind = c['kind']
        k = c['case']
        a = c['arm']
        xs = [
            t
            for t in ts
            if t['kind'] == kind and t['case'] == k and t['arm'] == a
        ]
        must(bool(xs), f'missing {kind}/{k}/{a}')
        if not xs:
            continue
        if kind == 'edithero':
            must(c['terminal'] or bool(c['error']), f'not finished {k}/{a}')
            must(
                all(x['scored'] == x['done'] for x in xs),
                f'unscored output {k}/{a}',
            )
        else:
            must(
                c['terminal']
                and xs[-1]['scored']
                and xs[-1]['passed'] is not None,
                f'not final/graded {k}/{a}',
            )
            must(sum(x['scored'] for x in xs) == 1, f'score count {k}/{a}')
            if xs[-1]['done']:
                study = Path(m['roots']['code'])
                n = next(
                    len(v['checklist'])
                    for v in map(
                        json.loads, (study / 'data/tasks.jsonl').open()
                    )
                    if v['case_id'] == k
                )
                must(
                    xs[-1]['items'] == n and xs[-1]['verifier_calls'] == n,
                    f'checklist count {k}/{a}',
                )
    budget = e['budget']
    limits = read(R / 'metrics/formal-limits.json')
    must(
        budget['reservations'] + budget['tools'] <= limits['calls'],
        'request limit',
    )
    must(budget['known_tokens'] < limits['tokens'], 'known token limit')
    must(
        budget['last_score'] - budget['start'] <= limits['seconds'],
        'time limit',
    )
    # Every new valid role receipt is visible at the delivered turn; interrupted requests may lack usage.
    accounted = sum(z['tokens'] for z in qs if z['source'] == R.name)
    must(
        accounted <= budget['known_tokens'],
        'more valid-actor tokens than new reservations',
    )
    out = dict(
        ready=not errors,
        errors=errors,
        assigned=len(cs),
        turns=len(ts),
        raw_new_attempts=budget['reservations'],
        attributable_new_tokens=accounted,
        known_reserved_tokens=budget['known_tokens'],
        missing_usage=budget['missing'],
    )
    (R / 'metrics/acceptance.json').write_text(json.dumps(out, indent=2))
    print(out['ready'], errors[:9], len(errors), 'gaps')
    return out


if __name__ == '__main__':
    check()
