"""Use actual invocation receipts, attached-image counts and first/last timestamps."""

from pathlib import Path
from collections import Counter
from prepare import R, read, save


def main():
    budget = read(R / 'metrics/formal-budget.json')
    counts = {}
    tech = []
    missing = []
    end = budget['start']
    for d in map(Path, budget['invocations']):
        recs = list(d.rglob('record.json'))
        diag = (R / 'diagnostics') in d.parents
        if diag:
            tech.append(str(d.relative_to(R)))
        if not recs:
            missing.append(str(d.relative_to(R)))
        for p in recs:
            v = read(p)
            if not v.get('usage'):
                if not diag:
                    missing.append(str(d.relative_to(R)))
                continue
            u = v['usage']
            end = max(end, p.stat().st_mtime)
            if diag:
                continue
            rel = d.relative_to(R)
            kind = rel.parts[0]
            stage = v['stage']
            arm = rel.parts[4] if kind == 'edithero' else rel.parts[3]
            key = f'{kind}/{arm}/{stage}'
            c = counts.setdefault(
                key,
                dict(
                    calls=0, input=0, output=0, cached=0, images=0, service_s=0
                ),
            )
            cmd = read(p.parent / 'command.json')
            c['images'] += sum(t in ['-i', '--image'] for t in cmd)
            c['calls'] += 1
            c['input'] += u['input_tokens']
            c['output'] += u['output_tokens']
            c['cached'] += u.get('cached_input_tokens', 0)
            c['service_s'] += v['latency_s']
    times = [
        p.stat().st_mtime
        for kind in ['edithero', '3dcodeedit']
        for p in (R / kind).rglob('score.json')
    ]
    save(
        R / 'metrics/billing.json',
        dict(
            reserved=len(budget['invocations']),
            start=budget['start'],
            last_request=end,
            last_score=max(times),
            elapsed_formal_until_last_score_s=max(times) - budget['start'],
            roles=counts,
            diagnostics=tech,
            missing_raw_receipts=missing,
        ),
    )


if __name__ == '__main__':
    main()
