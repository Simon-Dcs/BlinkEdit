"""One writer per arm/case. Resume offline edits at the reviewer, never repeat a delivered edit."""

import argparse, fcntl, os, subprocess, sys, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from prepare import R, read
from rules import eh, ce

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S))
from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.asset_edit import gltf

m = read(R / 'manifest.json')
sys.path.insert(0, m['roots']['code'])
from worker import review
DEFAULT_ROOT, DEFAULT_MANIFEST = R, m


def job(kind, key, arm, offline, gpu, *, root=None, manifest=None):
    R = root if root is not None else DEFAULT_ROOT
    m = manifest if manifest is not None else DEFAULT_MANIFEST
    group = R / kind / 'formal' / key / arm
    group.mkdir(parents=True, exist_ok=True)
    with (group / 'run.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'locked'
        n = (
            m['chain_info'][key]['n_turns']
            if kind == 'edithero'
            else m['max_revisions'] + 1
        )
        for turn in range(1, n + 1):
            out = group / f'turn{turn:02}'
            p = out / 'outcome.json'
            if not p.exists():
                if out.exists():
                    return 'orphan: ' + str(out)
                if kind == 'edithero':
                    root = R / 'setup/data'
                    source = root / 'states' / key / 'turn00.glb'
                    last = (
                        read(group / f'turn{turn - 1:02}' / 'outcome.json')
                        if turn > 1
                        else {}
                    )
                    if last:
                        source = Path(last['artifact'])
                    img = (
                        root
                        / 'renders'
                        / key
                        / f'turn{turn:02}'
                        / 'view0_cond.png'
                    )
                    if not (source.is_file() and (offline or img.is_file())):
                        return 'data'
                    q = m['chain_info'][key]['instructions'][turn - 1]
                    if offline and (
                        arm not in ('F0R', 'F1R')
                        or not eh(
                            q,
                            gltf.observe(gltf.header(source)[0])[0],
                            source,
                            last.get('retired_parts', {}),
                            last.get('created_parts', []),
                        )
                    ):
                        return 'model'
                elif offline:
                    pub = read(R / 'setup/public_code' / f'{key}.json')
                    src = (
                        Path(m['roots']['code'])
                        / 'data/sources'
                        / f"{pub['source_id']}.py"
                    )
                    if (
                        arm not in ('F0R', 'F1R')
                        or turn > 1
                        or not ce(pub['instruction'], src.read_text())
                    ):
                        return 'model'
                env = dict(
                    os.environ,
                    BENCH_ROOT=str(R),
                    BENCH_STAGE='formal',
                    BENCH_MONO=str(time.perf_counter()),
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    BENCH_RENDER_THREADS='4',
                    BENCH_RENDER_TIMEOUT='1200',
                )
                if offline:
                    env['BENCH_DEFER_REVIEW'] = '1'
                    env['BENCH_OFFLINE'] = '1'
                with (group / 'worker.log').open('a') as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(S / 'worker.py'),
                            str(R),
                            'formal',
                            kind,
                            key,
                            arm,
                            str(turn),
                        ],
                        env=env,
                        stdout=log,
                        stderr=log,
                        check=True,
                    )
            row = read(p)
            if row.get('review_pending'):
                if offline:
                    return 'model'
                t = time.perf_counter()
                try:
                    review(
                        out,
                        row,
                        read(R / 'setup/public_code' / f'{key}.json')[
                            'instruction'
                        ],
                        True,
                        '',
                    )
                except Exception as ex:
                    (out / 'review-error.txt').write_text(
                        traceback.format_exc()
                    )
                    return repr(ex)
                row.pop('review_pending')
                elapsed = time.perf_counter() - t
                row['elapsed_s'] += elapsed
                row['feedback_s'] += elapsed
                write_json(p, row)
            if row.get('error'):
                return 'error: ' + row['error']
            if kind == 'edithero' and not row['completed']:
                return 'blocked'
            if kind == '3dcodeedit' and row.get('accept'):
                return 'done'
        return 'done'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['offline', 'online'])
    parser.add_argument('kind', choices=['edithero', '3dcodeedit'])
    parser.add_argument('--workers', type=int, default=4)
    a = parser.parse_args()
    offline = a.mode == 'offline'
    os.environ.update(BENCH_ROOT=str(R), BENCH_STAGE='formal')
    if not offline and not (R / 'metrics/formal-limits.json').is_file():
        raise SystemExit('confirmed budget required')
    tasks = []
    for i, key in enumerate(
        m['formal']['edithero' if a.kind == 'edithero' else 'code']
    ):
        arms = m['arms']
        j = i % len(arms)
        for arm in arms[j:] + arms[:j]:
            tasks.append((a.kind, key, arm, offline, 6 + (i % 2)))
    with ThreadPoolExecutor(a.workers) as pool:
        jobs = {pool.submit(job, *args): args for args in tasks}
        for fut in as_completed(jobs):
            try:
                print(jobs[fut][:3], fut.result(), flush=True)
            except Exception:
                print(jobs[fut][:3], traceback.format_exc(), flush=True)


if __name__ == '__main__':
    main()
