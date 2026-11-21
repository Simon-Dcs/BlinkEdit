"""Extract one official chain from each uncompressed family tar; skip unrelated payloads by HTTP range."""

import argparse
import os
import tarfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def fetch(endpoint, revision, out, chain, kind):
    fam, name = chain.split('/')
    url = f'{endpoint}/datasets/AlayaLab/EditHero/resolve/{revision}/data/{kind}/{fam}.tar'
    client = requests.Session()
    client.mount(
        'https://',
        HTTPAdapter(
            max_retries=Retry(
                total=4,
                backoff_factor=1,
                status_forcelist=[429, 502, 503, 504],
            )
        ),
    )
    offset = 0
    prefix = f'data/{kind}/{chain}/'
    found = False
    while True:
        # Read a run of tar entries (payloads only when they belong to this chain).
        end = (
            offset
            + (8 * 1024**2 if found or kind != 'states' else 64 * 1024)
            - 1
        )
        with client.get(
            url,
            params={'slice': offset},
            headers={'Range': f'bytes={offset}-{end}'},
            stream=True,
            timeout=(20, 90),
        ) as response:
            response.raise_for_status()
            cr = response.headers.get('Content-Range', '')
            if response.status_code != 206 or not cr.startswith(
                f'bytes {offset}-'
            ):
                raise RuntimeError(f'Range not honored at {offset}: {cr}')
            block = response.content
        start = offset
        while offset + 512 <= start + len(block):
            raw = block[offset - start : offset - start + 512]
            if not any(raw):
                if not found:
                    raise ValueError(f'Missing {chain}')
                return
            info = tarfile.TarInfo.frombuf(raw, 'utf-8', 'strict')
            offset += 512
            if info.name.startswith(prefix) and info.isfile():
                found = True
                rel = Path(info.name)
                if '..' in rel.parts or rel.is_absolute():
                    raise ValueError(rel)
                path = out / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                n = min(info.size, len(block) - (offset - start))
                with path.with_suffix(path.suffix + '.partial').open(
                    'wb'
                ) as f:
                    f.write(block[offset - start : offset - start + n])
                    if n < info.size:
                        pos = offset + n
                        with client.get(
                            url,
                            params={'slice': pos},
                            headers={
                                'Range': f'bytes={pos}-{offset + info.size - 1}'
                            },
                            timeout=(20, 120),
                            stream=True,
                        ) as r:
                            r.raise_for_status()
                            if not r.headers.get(
                                'Content-Range', ''
                            ).startswith(f'bytes {pos}-'):
                                raise RuntimeError('range')
                            for chunk in r.iter_content(1 << 20):
                                f.write(chunk)
                    if f.tell() != info.size:
                        raise IOError('incomplete member')
                os.replace(path.with_suffix(path.suffix + '.partial'), path)
                print(path, info.size, flush=True)
            elif found and not info.name.startswith(prefix):
                return
            offset += (info.size + 511) // 512 * 512


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('revision')
    p.add_argument('out', type=Path)
    p.add_argument('chain')
    p.add_argument('kind')
    p.add_argument('--endpoint', default='https://huggingface.co')
    a = p.parse_args()
    fetch(a.endpoint, a.revision, a.out, a.chain, a.kind)
