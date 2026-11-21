"""Independently measure the delivered cabinet models; do not use these values for editing."""

from pathlib import Path
from design import R, read, write
from trace_call import locate
import numpy as np

KEY = 'openscad_small-parts-cabinet_L1_Q0'


def compare():
    pilot = R / 'setup/cabinet-pilot'
    old = {
        x['name']: x['bounds']
        for x in read(pilot / 'live-objects.json')['objects']
    }
    b = read(pilot / 'binding.json')
    names = [b['target_names'], b['related']]
    q = read(R / 'setup/public_code' / f'{KEY}.json')['instruction']
    summary = []
    for a in ('F1', 'F2', 'F2T'):
        base = R / '3dcodeedit/formal' / KEY / a
        ds = sorted(base.glob('turn*/edited.py'))
        for f in dict.fromkeys(ds[:1] + ds[-1:]):
            out = R / 'comparisons/cabinet' / a / f.parent.name
            out.mkdir(parents=True, exist_ok=True)
            if not (out / 'live-objects.json').is_file():
                locate(R, out, f.read_text(), q, '')
            if not (out / 'live-objects.json').is_file():
                continue
            new = {
                x['name']: x['bounds']
                for x in read(out / 'live-objects.json')['objects']
            }
            oth = set(old) - set(names[0]) - set(names[1])
            z = dict(
                arm=a,
                turn=f.parent.name,
                other_unchanged=sum(
                    k in new and np.allclose(old[k], new[k], atol=1e-4, rtol=0)
                    for k in oth
                ),
                other_n=len(oth),
                added=sorted(set(new) - set(old)),
                lost=sorted(set(old) - set(new)),
            )
            for g, ks in zip(['walls', 'liners'], names):
                z[g] = {
                    k: round(
                        (new[k][1][2] - new[k][0][2])
                        / (old[k][1][2] - old[k][0][2]),
                        4,
                    )
                    if k in new
                    else None
                    for k in ks
                }
            summary.append(z)
    write(R / 'comparisons/cabinet/geometry.json', summary)


if __name__ == '__main__':
    compare()
