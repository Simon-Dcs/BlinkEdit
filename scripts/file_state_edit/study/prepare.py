"""Freeze the two previously selected strata and fetch each archive only once."""

from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import tarfile

BASE = Path(__file__).resolve().parents[3] / 'outputs/file_state_edit'
R = BASE / '20261119_edit2'
OLD = BASE / '20261118_2200'
PROBE = BASE / '20261119_routing_probe'
SEL = BASE / '20261119_simple_selection/cases.json'


def read(p):
    return json.loads(p.read_text())


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')


def prepare():
    for sub in ['metrics', 'setup/public_code', 'comparisons', 'logs']:
        (R / sub).mkdir(parents=True, exist_ok=True)
    m = read(OLD / 'manifest.json')
    s = read(SEL)
    for f in [
        'ATTRIBUTION.csv',
        'LICENSE-DATA.md',
        'data/chains.json',
        'data/cameras.json',
    ]:
        p = R / 'setup' / f
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.symlink_to(OLD / 'setup' / f)
    cameras = read(R / 'setup/data/cameras.json')
    # argparse interprets negative numbers in scientific notation as option names.
    # Decimal string formatting changes no camera coordinate.
    for camera in cameras.values():
        c = camera['frame']
        c['frame_center'] = [
            format(Decimal(str(x)), 'f') for x in c['frame_center']
        ]
    save(R / 'setup/render-rig/cameras.json', cameras)
    simple = {
        'edithero': [v['chain'] for v in s['edithero']],
        'code': [v['case_id'] for v in s['code']],
    }
    m['simple'] = simple
    m['general'] = m['formal']
    m['formal'] = {k: simple[k] + m['general'][k] for k in simple}
    m['focus'] = {v['chain']: v['turn'] for v in s['edithero']}
    m['known_examples'] = [s['edithero'][0]['chain'], s['code'][0]['case_id']]
    m['arms'] = ['F0', 'F0R', 'F1R']
    m['max_revisions'] = 8
    m['study'] = 'edit2'
    m['final_only'] = True
    d = {v['chain']: v for v in read(R / 'setup/data/chains.json')}
    m['chain_info'] = {k: d[k] for k in m['formal']['edithero']}
    for t in map(
        json.loads, (Path(m['roots']['code']) / 'data/tasks.jsonl').open()
    ):
        if t['case_id'] in m['formal']['code']:
            save(
                R / 'setup/public_code' / f"{t['case_id']}.json",
                {k: t[k] for k in ['source_id', 'instruction']},
            )
    save(R / 'manifest.json', m)
    remaining = {}
    for kind in ['states', 'renders', 'part_masks']:
        for key in m['formal']['edithero']:
            p = R / 'setup/data' / kind / key
            p.parent.mkdir(parents=True, exist_ok=True)
            for d in [OLD / 'setup/data', PROBE / 'data']:
                source = d / kind / key
                if source.is_dir():
                    if not p.exists():
                        p.symlink_to(source, target_is_directory=True)
                    break
            else:
                remaining.setdefault((kind, key.split('/')[0]), set()).add(key)
    return m, remaining


def fetch_group(job):
    (kind, fam), keys = job
    m = read(R / 'manifest.json')
    out = R / 'setup'
    rev = m['dataset_revision']
    url = f'https://hf-mirror.com/datasets/AlayaLab/EditHero/resolve/{rev}/data/{kind}/{fam}.tar'
    ses = requests.Session()
    ses.mount(
        'https://',
        HTTPAdapter(
            max_retries=Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=(429, 502, 503, 504),
            )
        ),
    )

    def get(off, end):
        resp = ses.get(
            url,
            params={'slice': off},
            headers={'Range': f'bytes={off}-{end}'},
            timeout=(30, 150),
        )
        resp.raise_for_status()
        if resp.status_code != 206 or not resp.headers.get(
            'Content-Range', ''
        ).startswith(f'bytes {off}-'):
            raise ValueError(f'invalid range {off}')
        return resp.content

    index = R / 'logs' / f'fetch-{kind}-{fam}.json'
    state = (
        read(index) if index.exists() else dict(offset=0, seen=[], last=None)
    )
    off = state['offset']
    done = set(state['seen'])
    last = state['last']
    while keys - done:
        start = off
        chunk = get(
            off,
            off
            + (65536 if kind == 'states' and last not in keys else 1048576)
            - 1,
        )
        while off + 512 <= start + len(chunk):
            raw = chunk[off - start : off - start + 512]
            if not any(raw):
                if last in keys:
                    done.add(last)
                if keys - done:
                    raise IOError(f'Cannot find {keys - done}')
                break
            info = tarfile.TarInfo.frombuf(raw, 'utf-8', 'strict')
            off += 512
            parts = Path(info.name).parts
            key = '/'.join(parts[2:4]) if len(parts) >= 4 else None
            if last != key and last in keys:
                done.add(last)
            last = key
            if key in keys and info.isfile():
                if Path(info.name).is_absolute() or '..' in parts:
                    raise ValueError(info.name)
                path = out / info.name
                if not path.exists() or path.stat().st_size != info.size:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pos = off
                    part = path.with_suffix(path.suffix + '.partial')
                    with part.open('wb') as f:
                        head = chunk[off - start : off - start + info.size]
                        f.write(head)
                        pos += len(head)
                        while pos < off + info.size:
                            block = get(
                                pos,
                                min(
                                    off + info.size - 1, pos + 8 * 1024**2 - 1
                                ),
                            )
                            block = block[: off + info.size - pos]
                            if not block:
                                raise IOError('short range')
                            f.write(block)
                            pos += len(block)
                    os.replace(part, path)
            off += (info.size + 511) // 512 * 512
            if not keys - done:
                break
        save(index, dict(offset=off, seen=sorted(done), last=last))
    return kind, fam


if __name__ == '__main__':
    m, jobs = prepare()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for ret in pool.map(fetch_group, jobs.items()):
            print(ret, flush=True)
