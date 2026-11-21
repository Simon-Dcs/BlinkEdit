"""Frozen, serial paired schedule with resource and request limits."""
import argparse
import importlib.metadata
import json
import os
import random
import select
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common import TASKS, VERSION, append, save


class Benchmark:
    def __init__(self, root, phase):
        self.root, self.phase = root.resolve(), phase
        self.resident = None
        self.resident_start = None
        self.resident_log = None
        self.started = time.perf_counter()
        past = [json.loads(line) for line in (root/'metrics/requests.jsonl').read_text().splitlines()] if (root/'metrics/requests.jsonl').exists() else []
        self.requests = len(past)
        self.tokens = sum((r.get('usage') or {}).get('input_tokens',0)+(r.get('usage') or {}).get('output_tokens',0) for r in past)
        self.consecutive_failures = 0
        self.previous_formal_s = sum(json.loads(f.read_text()).get('wall_s',0) for f in root.glob('*_consumption.json')
                                    if not f.stem.startswith('pilot'))
        self.records = []

    def check_limits(self):
        records = self.root/'metrics/tasks.jsonl'
        if records.exists() and len(records.read_text().splitlines()) >= 78:
            raise RuntimeError('78 timed-round limit reached')
        if self.requests >= 100 or self.tokens >= 2000000 or self.previous_formal_s+time.perf_counter()-self.started >= 7200:
            raise RuntimeError('Experiment resource threshold reached; do not start another task')
        if self.consecutive_failures >= 3:
            raise RuntimeError('Three consecutive model failures; affected stage stopped')
        gpu = subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        allowed = {str(self.resident.pid)} if self.resident else set()
        if any(p.strip() not in allowed for p in gpu.splitlines() if p.strip()):
            raise RuntimeError('GPU 7 conflict; no other GPU is permitted')

    def run(self, task, condition, repeat, input_path=None, history=None, session_id=None, turn_id=1, video=False):
        self.check_limits()
        task_id = f'{self.phase}_{condition}_{task}_r{repeat}'
        d = self.root/'runs'/condition/task_id
        if d.exists():
            raise RuntimeError(f'Output already exists, refusing overwrite: {d}')
        d.mkdir(parents=True)
        job = dict(run_id=self.root.name, task_id=task_id, task=task, condition=condition, phase=self.phase,
                   session_id=session_id, turn_id=turn_id, seed=41+repeat, instruction=TASKS[task],
                   input=str(input_path or self.root/'inputs/two_rooms/scene.urdf'),
                   history=history or [], directory=str(d), assets=str(self.root/'inputs/source'), video=video,
                   max_model_requests=min(2,100-self.requests))
        job_path = d/'job.json'
        save(job_path, job)
        start = time.perf_counter()
        job['start_mono'] = start
        save(job_path, job)
        if condition in ('M2','M3'):
            if self.resident is None:
                self.resident_start = start
                self.resident_log = (self.root/f'{self.phase}_resident.log').open('w')
                self.resident = subprocess.Popen([sys.executable, 'scripts/efficiency_optimization/worker.py',
                    '--resident','--reply-dir',str(self.root/'resident')], stdin=subprocess.PIPE,
                    stdout=self.resident_log, stderr=subprocess.STDOUT, text=True)
            self.resident.stdin.write(json.dumps(job, ensure_ascii=False)+'\n')
            self.resident.stdin.flush()
            reply = self.root/'resident'/f'{task_id}.json'
            while not reply.exists():
                if self.resident.poll() is not None:
                    raise RuntimeError('Resident worker exited; see resident log')
                if time.perf_counter()-start > 600:
                    # Only our own benchmark worker is affected.
                    self.resident.terminate()
                    raise RuntimeError('Resident task timeout 600s')
                time.sleep(.05)
        else:
            with (d/'console.log').open('w') as log:
                proc = subprocess.run([sys.executable, 'scripts/efficiency_optimization/worker.py', '--job',str(job_path)],
                                      stdout=log, stderr=subprocess.STDOUT, timeout=600)
            if proc.returncode:
                raise RuntimeError(f'Cold worker failed at {d}; see console.log')
        rec = json.loads((d/'result.json').read_text())
        rec['controller_complete_mono'] = time.perf_counter()
        rec['verified_s'] = rec['verified_mono']-start
        rec['edit_s'] = rec.get('edit_end_mono',rec['verified_mono'])-start
        rec['process_overhead_s'] = rec['received_mono']-start
        rec['controller_wall_s'] = rec['controller_complete_mono']-start
        if condition in ('M2','M3'):
            rec['resident_first'] = not any(r['condition']==condition for r in self.records)
        rec['directory'] = str(d)
        save(d/'result.json', rec)
        append(self.root/'metrics/tasks.jsonl', rec)
        for request in rec['requests']:
            append(self.root/'metrics/requests.jsonl',request)
            self.consecutive_failures = self.consecutive_failures+1 if request['status']!='ok' else 0
        for event in rec['events']:
            append(self.root/'metrics/stages.jsonl', dict(task_id=task_id, condition=condition, **event))
        self.requests += len(rec['requests'])
        self.tokens += rec['known_tokens']
        self.records.append(rec)
        print(task_id, rec['status'], 'PASS' if rec['accepted'] else 'FAIL',
              f"{rec['verified_s']:.2f}s", rec['known_tokens'], f'cumulative {self.requests} requests / {self.tokens} tokens', flush=True)
        if not rec['accepted'] and (not rec.get('validation',{}).get('unchanged_poses_and_meshes',True)
                                   or 'state/inventory mismatch' in rec.get('error','')):
            # Preserve failed snapshots. Do not accumulate misleading speed data
            # after a broken candidate or stale-state incident.
            save(self.root/'STOPPED.json', dict(task_id=task_id, reason=rec.get('error','acceptance failure')))
            raise RuntimeError(f'State contamination: {task_id}; investigate before another task')
        return rec

    def close(self):
        if self.resident and self.resident.poll() is None:
            self.resident.stdin.write('{"shutdown":true}\n')
            self.resident.stdin.flush()
            self.resident.wait(timeout=60)
        if self.resident_start:
            save(self.root/f'{self.phase}_resident_lifetime.json', dict(start=self.resident_start,
                end=time.perf_counter(), wall_s=time.perf_counter()-self.resident_start,
                note='Includes interleaved waiting; use task phase sums for amortized active cost.'))
        if self.resident_log:
            self.resident_log.close()
        save(self.root/f'{self.phase}_consumption.json',dict(requests=self.requests, known_tokens=self.tokens,
            wall_s=time.perf_counter()-self.started))


def setup():
    from shutil import copytree, copy2
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    auth = subprocess.run(['codex','login','status'],capture_output=True,text=True)
    assert 'Logged in using ChatGPT' in auth.stdout+auth.stderr
    root = Path('outputs/efficiency_optimization')/datetime.now().strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True)
    for child in ['metrics','inputs','comparisons','resident','source_snapshot','setup']:
        (root/child).mkdir()
    old = Path('outputs/efficiency/20261112_214800')
    t = time.perf_counter()
    copytree(old/'fixtures/two_rooms',root/'inputs/two_rooms')
    copytree(old/'source',root/'inputs/source')
    copy2(old/'manifest.json',root/'inputs/previous_manifest.json')
    copy2('efficiency_plan.md',root/'source_snapshot/efficiency_plan.md')
    for file in Path('scripts/efficiency_optimization').glob('*.py'):
        copy2(file,root/'source_snapshot'/file.name)
    schedule = []
    for repeat in range(1,4):
        for task in ['T1','T2','T3','T4']:
            groups = ['M0','M1']+(['M2'] if task in ['T3','T4'] else [])
            # Rotate deterministically; every condition gets early and late positions.
            shift = (repeat-1+int(task[1:]))%len(groups)
            groups = groups[shift:]+groups[:shift]
            schedule.extend(dict(task=task,condition=c,repeat=repeat) for c in groups)
    for repeat in range(1,4):
        for condition in (['M0','M1'] if repeat%2 else ['M1','M0']):
            schedule.extend(dict(task=f'C{i}',condition=condition,repeat=repeat) for i in range(1,5))
    for i,task in enumerate(['B1','B2','B3']):
        schedule.extend(dict(task=task,condition=c,repeat=1) for c in (['M0','BEST'] if i%2==0 else ['BEST','M0']))
    manifest = dict(version=VERSION, created_at=datetime.now().astimezone().isoformat(),
        repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy','trimesh','sapien','openai','shapely']},
        codex_version=subprocess.check_output(['codex','--version'],text=True).strip(),
        login_status=auth.stdout+auth.stderr, model='gpt-6-astra', effort='medium', gpu='7',
        source_manifest=str(old/'manifest.json'), source_dataset_revision='58258b50a0fc95034f2f3cc03b332ec6f72b91fd',
        setup_copy_s=time.perf_counter()-t, tasks=TASKS, schedule=schedule,
        limits=dict(rounds=78,requests=100,known_tokens=2000000,formal_seconds=7200,request_timeout=180,task_timeout=600),
        acceptance=dict(translation_m=.02,rotation_deg=5,simulation_seconds=2,friction=[.6,.5],contact_offset_m=.001,
                        views=2,resolution=[720,1080],mesh_samples=10000,max_attempts=2000,rotation_rpy=[1.5708,0,0]),
        seed_policy='42,43,44; native trimesh surface sampling independently entropy-seeded, not bitwise paired',
        thresholds=dict(P1_session_tokens_reduction=.10,P2_amortized_verified_reduction=.15),
        machine_shared=True, allocation='No parallel model calls, native CPU edits, SAPIEN rendering on physical GPU7')
    save(root/'manifest.json',manifest)
    for cmd,name in [(['nvidia-smi'],'nvidia-smi.txt'),([sys.executable,'-m','pip','freeze'],'pip-freeze.txt'),
                     ([sys.executable,'-m','pip','check'],'pip-check.txt'),(['conda','list'],'conda-list.txt')]:
        r = subprocess.run(cmd,capture_output=True,text=True)
        (root/'setup'/name).write_text(r.stdout+r.stderr)
    Path('outputs/efficiency_optimization/latest_run.txt').write_text(str(root.resolve())+'\n')
    print(root.resolve())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=['setup','pilot','pilot_finish','formal','combination'])
    p.add_argument('--root')
    args = p.parse_args()
    if args.phase == 'setup':
        setup()
        return
    root = Path(args.root or Path('outputs/efficiency_optimization/latest_run.txt').read_text().strip())
    b = Benchmark(root,args.phase)
    try:
        if args.phase == 'pilot':
            # One cold full edit plus two resident resets, including a no-op boundary.
            b.run('T4','M0',1,video=True)
            b.run('T3','M2',1)
            b.run('B2','M2',1)
        elif args.phase == 'pilot_finish':
            b.run('B2','M0',2)
            b.run('B3','M1',2)
            b.run('T4','M1',2)
        elif args.phase == 'formal':
            history, inputs = {}, {}
            schedule = json.loads((root/'manifest.json').read_text())['schedule']
            for item in schedule:
                task, c, repeat = item['task'], item['condition'], item['repeat']
                if c == 'BEST':
                    # Best candidate on the already-collected main data, not on boundary outcomes.
                    def avg(group):
                        rows = [r for r in b.records if r['condition']==group and r['task'] in ['T3','T4']]
                        return sum(r['verified_s'] for r in rows)/len(rows)
                    c = min(['M1','M2'],key=avg)
                    save(root/'boundary_candidate.json',dict(condition=c,rule='Lowest mean verified_s on T3/T4, including first startup'))
                sid = f'{c}_session_r{repeat}' if task.startswith('C') else None
                rec = b.run(task,c,repeat,input_path=inputs.get(sid),history=history.get(sid),session_id=sid,
                            turn_id=int(task[1:]) if sid else 1,
                            video=repeat==1 and task=='T4')
                if sid:
                    history[sid], inputs[sid] = rec['history'],Path(rec['output'])
        else:
            for repeat in range(1,4):
                for task in ['T3','T4']:
                    for c in (['M0','M3'] if repeat%2 else ['M3','M0']):
                        b.run(task,c,repeat)
    finally:
        b.close()


if __name__ == '__main__':
    main()
