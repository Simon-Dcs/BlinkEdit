"""Sequential, bounded benchmark of the project's documented CLI paths."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--only', nargs='*')
p.add_argument('--repeats', type=int, default=3)
args = p.parse_args()
root = Path('outputs/efficiency/latest_run.txt').read_text().strip()
root = Path(root)
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
assert os.environ.get('GPT_PROVIDER') == 'codex'
auth = subprocess.run(['codex','login','status'], capture_output=True, text=True)
assert 'Logged in using ChatGPT' in auth.stdout+auth.stderr
cases = ['S1_delete_semantic', 'S2_insert_semantic', 'H1_four_semantic',
         'A1_four_exact_sequential', 'A2_four_exact_batch', 'N1_impossible']
if args.only:
    cases = args.only


def run_call(case_dir, seed, index, cfg):
    d = case_dir/f'call_{index:02d}'
    d.mkdir()
    cmd = [sys.executable, 'scripts/efficiency/trace_cli.py', '--run-dir', str(d), '--seed', str(seed), *cfg]
    (d/'command.json').write_text(json.dumps(cmd, ensure_ascii=False, indent=2))
    t = time.perf_counter()
    with (d/'console.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    rec = dict(seconds=time.perf_counter()-t, returncode=result.returncode, directory=str(d))
    (d/'process.json').write_text(json.dumps(rec, indent=2))
    return rec


for i in range(args.repeats):
    seed = 42+i
    for case in cases:
        d = root/'runs'/f'{case}_r{i+1}'
        if d.exists():
            print('EXISTS skip', d.name, flush=True)
            continue
        d.mkdir()
        tcopy = time.perf_counter()
        shutil.copytree(root/'fixtures/two_rooms', d/'scene')
        prep_s = time.perf_counter()-tcopy
        src = d/'scene/scene.urdf'
        t = time.perf_counter()
        rec = dict(case=case, repeat=i+1, seed=seed, started_at=datetime.now().astimezone().isoformat(), calls=[], input_copy_seconds=prep_s)
        if case == 'E1_full_request':
            request = '移除办公室地上的台灯，然后在办公室的桌子上放一盏台灯、一个花瓶、一本书和一个杯子，物体不要互相重叠；餐厅保持不变。'
            rec['request'] = request
            cfg = ['--plan-request', request, '--asset-root', str(root/'source'),
                   '--urdf_path', str(src), '--output_path', str(d/'floorplan.png')]
            rec['calls'].append(run_call(d, seed, 0, cfg))
        elif case == 'S1_delete_semantic':
            cfg = ['--urdf_path', str(src), '--delete_instance', '办公室里的台灯', '--output_path', str(d/'floorplan.png')]
            rec['calls'].append(run_call(d, seed, 0, cfg))
        elif case in ['S2_insert_semantic', 'N1_impossible']:
            asset = 'vase' if case.startswith('S2') else 'sofa'
            cfg = ['--urdf_path', str(src), '--asset_path', str(root/f'source/{asset}/mesh/{asset}.obj'), '--instance_key', f'new_{asset}', '--on_instance', '办公室的桌子', '--in_room', '办公室', '--place_strategy', 'top', '--rotation_rpy', '1.5708', '0', '0', '--output_path', str(d/'floorplan.png')]
            rec['calls'].append(run_call(d, seed, 0, cfg))
        else:
            objects = [dict(asset_path=str(root/f'source/{a}/mesh/{a}.obj'), instance_key=f'new_{a}', on_instance='table_office_001', in_room='office_floor', rotation_rpy=[1.5708, 0, 0], place_strategy='top') for a in ['lamp','vase','book','mug']]
            # Retain canonical workload outside the scene; CLI temp is removed.
            (d/'workload.json').write_text(json.dumps(objects, indent=2))
            if case == 'A2_four_exact_batch':
                tmp = d/'scene/batch.json'
                tmp.write_text(json.dumps(objects))
                try:
                    rec['calls'].append(run_call(d, seed, 0, ['--urdf_path', str(src), '--batch_insert_config', str(tmp), '--output_path', str(d/'floorplan.png')]))
                finally:
                    tmp.unlink(missing_ok=True)
            else:
                for j, obj in enumerate(objects):
                    semantic = case == 'H1_four_semantic'
                    cfg = ['--urdf_path', str(src), '--asset_path', obj['asset_path'], '--instance_key', obj['instance_key'], '--on_instance', '办公室的桌子' if semantic else obj['on_instance'], '--in_room', '办公室' if semantic else obj['in_room'], '--place_strategy', 'top', '--rotation_rpy', '1.5708', '0', '0']
                    if j == len(objects)-1:
                        cfg += ['--output_path', str(d/'floorplan.png')]
                    call = run_call(d, seed+j, j, cfg)
                    rec['calls'].append(call)
                    if call['returncode'] != 0:
                        break
                    src = d/'scene/scene_updated.urdf'
        rec.update(seconds=time.perf_counter()-t, ended_at=datetime.now().astimezone().isoformat(), cli_success=all(x['returncode']==0 for x in rec['calls']))
        (d/'case.json').write_text(json.dumps(rec, indent=2))
        with (root/'metrics/cases.jsonl').open('a') as f:
            f.write(json.dumps(rec)+'\n')
        print(d.name, rec['cli_success'], round(rec['seconds'],3), flush=True)
