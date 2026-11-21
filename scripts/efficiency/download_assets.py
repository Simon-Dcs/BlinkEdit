"""Download only official demo OBJ/collision/material/URDF files; no weights."""
import json
import time
from pathlib import Path
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

root = Path('outputs/efficiency/latest_run.txt').read_text().strip()
root = Path(root)
start = time.perf_counter()
api = 'https://huggingface.co/api/datasets/HorizonRobotics/EmbodiedGenData'
revision = requests.get(api, timeout=60).json()['sha']
records = []
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
for name in ['table', 'vase', 'book', 'mug', 'lamp']:
    prefix = f'demo_assets/{name}/result'
    listing = requests.get(f'{api}/tree/{revision}/{prefix}', params={'recursive': 'true'}, timeout=60)
    listing.raise_for_status()
    for entry in listing.json():
        path = entry['path']
        if entry['type'] != 'file' or Path(path).suffix not in ['.urdf', '.obj', '.mtl', '.png']:
            continue
        url = f'https://huggingface.co/datasets/HorizonRobotics/EmbodiedGenData/resolve/{revision}/{path}'
        dest = root/'source'/name/Path(path).relative_to(prefix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        cached = dest.exists() and dest.stat().st_size == entry['size']
        if not cached:
            response = session.get(url, timeout=120)
            response.raise_for_status()
            dest.write_bytes(response.content)
        record = dict(url=url, path=str(dest), bytes=dest.stat().st_size, seconds=time.perf_counter()-t, cached=cached)
        records.append(record)
        with (root/'setup/download_files.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(path, dest.stat().st_size, flush=True)
(root/'setup/downloads.json').write_text(json.dumps(dict(revision=revision, ended_at=datetime.now().astimezone().isoformat(), seconds=time.perf_counter()-start, files=records), indent=2))
