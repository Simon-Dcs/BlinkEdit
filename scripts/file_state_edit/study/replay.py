"""Matched local timing for the longest available bilateral local-only prefix of each chain."""

import statistics, tempfile, sys
from pathlib import Path
from prepare import R, read, save

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_cost import replay


def main():
    m = read(R / 'manifest.json')
    out = []
    for key in m['formal']['edithero']:
        g = R / 'edithero/formal' / key
        candidates = []
        for i in range(1, m['chain_info'][key]['n_turns'] + 1):
            ds = [g / a / f'turn{i:02}' for a in ['F0R', 'F1R']]
            if any(
                not (d / 'outcome.json').exists()
                or not read(d / 'outcome.json').get('completed')
                or read(d / 'outcome.json').get('route') != 'local'
                for d in ds
            ):
                break
            v = [read(d / 'actions.json') for d in ds]
            if v[0] != v[1] or any(a['op'] == 'restore' for a in v[0]):
                break
            candidates.append(ds[0] / 'actions.json')
        if not candidates:
            continue
        with tempfile.TemporaryDirectory(dir=R) as td:
            h = Path(td)
            (h / 'setup').symlink_to(R / 'setup', target_is_directory=True)
            d = h / 'edithero/dev' / key / 'F0'
            for i, p in enumerate(candidates, 1):
                t = d / f'turn{i:02}'
                t.mkdir(parents=True)
                (t / 'actions.json').symlink_to(p)
            samples = {a: [] for a in ['F0', 'F1']}
            for i in range(12):
                for arm in ['F0', 'F1'] if i % 2 == 0 else ['F1', 'F0']:
                    with tempfile.TemporaryDirectory(dir=h) as dest:
                        samples[arm].append(replay(h, key, arm, Path(dest)))
            out.append(
                dict(
                    chain=key,
                    turns=len(candidates),
                    median_s={
                        a: statistics.median(x['seconds'] for x in v)
                        for a, v in samples.items()
                    },
                    samples=samples,
                )
            )
    save(R / 'metrics/matched-local.json', out)


if __name__ == '__main__':
    main()
