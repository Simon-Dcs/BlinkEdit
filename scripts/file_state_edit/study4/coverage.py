"""Characterize exactly where each frozen first-edit rule would route the 24 inputs."""

from pathlib import Path
import sys
from design import R, P, read, write
from trace_call import locate
from embodied_gen.harness.asset_edit import local_program

sys.path.insert(0, str(P))
from study import rules


def run():
    m = read(R / 'manifest.json')
    xs = []
    for k in m['formal']['code']:
        task = read(R / 'setup/public_code' / f'{k}.json')
        q = task['instruction']
        src = (
            Path(m['roots']['code'])
            / 'data/sources'
            / f"{task['source_id']}.py"
        ).read_text()
        group = (
            'fresh'
            if k in m['fresh']['code']
            else 'simple'
            if k in m['simple']['code']
            else 'general'
        )
        if rules.ce(q, src):
            xs.append(dict(case=k, group=group, F2='rule', F2T='rule'))
            continue
        d = R / 'comparisons/first-routing' / k
        d.mkdir(parents=True, exist_ok=True)
        active = R / '3dcodeedit/formal' / k / 'F2T/turn01/binding.json'
        prior = d / 'binding.json'
        if active.is_file():
            b = read(active)
        elif prior.is_file():
            b = read(prior)
        else:
            _, b = locate(R, d, src, q, '')
        xs.append(
            dict(
                case=k,
                group=group,
                F2='patch' if local_program.windows(src, q, '') else 'full',
                F2T='patch' if b['spans'] else 'full',
                reason=b['reason'],
                objects=b.get('objects'),
                mapped=b.get('mapped'),
                parts=b.get('target_names', []),
            )
        )
    write(R / 'comparisons/first-routing.json', xs)


if __name__ == '__main__':
    run()
