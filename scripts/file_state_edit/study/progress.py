"""Report actual coverage and failure reasons; never count unstarted arms as zero-cost methods."""

import collections, sys
from prepare import R, read


def status():
    m = read(R / 'manifest.json')
    for kind, tag in [('edithero', 'edithero'), ('3dcodeedit', 'code')]:
        rows = []
        for key in m['formal'][tag]:
            for arm in m['arms']:
                p = R / kind / 'formal' / key / arm
                vals = [read(v) for v in sorted(p.glob('turn*/outcome.json'))]
                if not vals:
                    continue
                x = vals[-1]
                finished = (
                    x['turn'] == m['chain_info'][key]['n_turns']
                    if kind == 'edithero'
                    else x.get('accept') or x['turn'] > m['max_revisions']
                )
                rows.append(
                    (
                        key,
                        arm,
                        len(vals),
                        bool(finished),
                        x.get('error'),
                        len(list(p.glob('turn*/score.json'))),
                    )
                )
        print(
            kind,
            dict(
                assigned=3 * len(m['formal'][tag]),
                started=len(rows),
                process_complete=sum(int(i[3]) for i in rows),
                stopped_with_error=sum(bool(i[4]) for i in rows),
                edits=sum(i[2] for i in rows),
                scored=sum(i[5] for i in rows),
            ),
        )
        for x in rows:
            if x[4] and (
                'Codex' in x[4] or 'Budget' in x[4] or 'Transport' in x[4]
            ):
                print('Transport issue', x[:5])
    print(
        'reserved', len(read(R / 'metrics/formal-budget.json')['invocations'])
    )


if __name__ == '__main__':
    status()
