"""One-shot or JSON-lines resident execution worker. stdout is a file protocol."""
import argparse
import importlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

BOOT = time.perf_counter()
from common import Meter, VERSION, check_plan, make_prompt, save
import numpy as np
from validation import Renderer, validate

api = importlib.import_module('embodied_gen.skills.spatial-computing.api.floorplan_api')
col = importlib.import_module('embodied_gen.skills.spatial-computing.core.collector')
gc = importlib.import_module('embodied_gen.utils.gpt_clients')


class Runtime:
    def __init__(self):
        self.meter = Meter()
        self.meter.install(gc, api, col)
        self.manager = None
        self.renderer = None

    def load(self, path):
        if self.manager is None:
            self.manager = api.FloorplanManager(urdf_path=str(path))
        else:
            self.manager.urdf_path = str(path)
            self.manager.pending_instance_data = None
            self.manager.collector.collect(str(path))
        return self.manager

    def run(self, job):
        m = self.meter
        m.events, m.stack, m.requests = [], [], []
        d = Path(job['directory'])
        d.mkdir(parents=True, exist_ok=True)
        m.directory = d
        m.identity = {k: job[k] for k in ['run_id','task_id','session_id','turn_id','condition','task','phase']}
        started = time.perf_counter()
        random.seed(job['seed'])
        np.random.seed(job['seed'])
        rec = dict(**m.identity, version=VERSION, seed=job['seed'], start_mono=job['start_mono'],
                   received_mono=started, worker_pid=os.getpid(), bootstrap_s=started-BOOT,
                   accepted=False, status='failed', history=job.get('history', []), instruction=job['instruction'])
        frames = []
        try:
            with m.span('input_copy_and_reset'):
                if self.renderer is not None:
                    self.renderer.reset()
                original = Path(job['input'])
                scene = d/'candidate'
                shutil.copytree(original.parent, scene)
                source = scene/original.name
            catalog = {a: str(Path(job['assets'])/a/'mesh'/f'{a}.obj')
                       for a in ['lamp','vase','book','mug','sofa']}
            with m.span('context_construction'):
                manager = self.load(source)
                names = sorted(manager.get_instance_names())
                rooms = sorted(manager.get_room_names())
                # Guard independent-task reset against objects leaking from earlier requests.
                import xml.etree.ElementTree as ET
                tree_names = {link.get('name').lower() for link in ET.parse(source).findall('link')
                              if link.find('visual') is not None and 'floor' not in link.get('name')}
                if set(names) != tree_names:
                    raise RuntimeError(f'Native state/inventory mismatch: {names}, {tree_names}')
                prompt = make_prompt(job['condition'], names, rooms, catalog, job.get('history', []), job['instruction'])
                (d/'planner_input.txt').write_text(prompt)
            error = None
            for attempt in range(min(2,job.get('max_model_requests',2))):
                with m.span('planning', attempt=attempt+1):
                    query = prompt if error is None else prompt+'\nThe previous plan failed legality checks: '+error+'\nReturn one corrected full JSON plan.'
                    reply = gc.GPT_CLIENT.query(query)
                if reply is None:
                    raise RuntimeError('Model request failed; no blind retries')
                try:
                    with m.span('plan_legality', attempt=attempt+1):
                        plan = check_plan(json.loads(reply), names, rooms, catalog)
                    break
                except (ValueError, TypeError, KeyError) as exc:
                    error = str(exc)
                    save(d/f'plan_error_{attempt}.json', dict(error=error, reply=reply))
                    if attempt:
                        raise
            save(d/'plan.json', plan)
            status = plan['status']
            after = source
            with m.span('native_edit'):
                if status == 'execute':
                    for key in plan['delete']:
                        if not manager.delete_object(key):
                            raise RuntimeError(f'Native deletion failed: {key}')
                    configs = [dict(asset_path=catalog[i['asset']], instance_key=i['instance_key'],
                                    on_instance=i['on_instance'], in_room=i['in_room'], place_strategy='top',
                                    rotation_rpy=[1.5708,0,0], n_max_attempt=2000) for i in plan['insert']]
                    centers = manager.batch_insert_objects(configs) if configs else []
                    save(d/'native_result.json', dict(configs=configs, centers=centers))
                    if any(c is None for c in centers):
                        # Never publish partial native writes. Deterministic infeasibility is
                        # returned once, rather than re-prompting until constraints disappear.
                        status = 'impossible'
                        rec['native_rejection'] = True
                        with m.span('rollback'):
                            manager = self.load(source)
                        after = source
                    else:
                        after = Path(manager._get_output_path(str(source)))
                manager.visualize(str(d/'floorplan.png'))
            rec['edit_end_mono'] = time.perf_counter()
            with m.span('verification'):
                if self.renderer is None:
                    with m.span('renderer_initialization'):
                        self.renderer = Renderer()
                result, frames = validate(job['task'], original, after, status, self.renderer, d, m,
                                           video=job.get('video',False))
            rec.update(status=status, accepted=result['accepted'], validation=result, output=str(after),
                       original=str(original), plan=plan)
            if result['accepted']:
                with m.span('publish'):
                    save(d/'published.json', dict(urdf=str(after), status=status))
                    # Failed/no-op/clarification records carry no phantom changes into history.
                    h = dict(instruction=job['instruction'], status=status,
                             deleted=plan['delete'] if status=='execute' else [],
                             inserted=plan['insert'] if status=='execute' else [])
                    rec['history'] = job.get('history', [])+[h]
            else:
                # Only accepted output can become the next turn's real state.
                rec['output'] = str(original)
        except BaseException as exc:
            rec['error'] = repr(exc)
            traceback.print_exc()
        rec['verified_mono'] = time.perf_counter()
        rec['worker_task_s'] = rec['verified_mono']-started
        rec['requests'] = m.requests
        rec['events'] = m.events
        rec['known_tokens'] = sum((r.get('usage') or {}).get('input_tokens',0)+(r.get('usage') or {}).get('output_tokens',0) for r in m.requests)
        rec['unknown_usage_requests'] = sum(r.get('usage') is None for r in m.requests)
        # Encoding is deliberately beyond the common acceptance endpoint.
        if frames:
            t = time.perf_counter()
            import imageio.v2 as imageio
            imageio.mimwrite(d/'release.mp4',frames,fps=30,macro_block_size=1)
            rec['video_encoding_s'] = time.perf_counter()-t
        save(d/'result.json', rec)
        return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job')
    p.add_argument('--resident', action='store_true')
    p.add_argument('--reply-dir')
    args = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'
    assert os.environ['GPT_PROVIDER'] == 'codex'
    assert os.environ['MODEL_NAME'] == 'gpt-6-astra'
    runtime = Runtime()
    if args.job:
        runtime.run(json.loads(Path(args.job).read_text()))
    else:
        reply_dir = Path(args.reply_dir)
        save(reply_dir/'ready.json', dict(pid=os.getpid(), bootstrap_s=time.perf_counter()-BOOT,
                                        ready_mono=time.perf_counter()))
        for line in sys.stdin:
            job = json.loads(line)
            if job.get('shutdown'):
                break
            result = runtime.run(job)
            save(reply_dir/f"{job['task_id']}.json", dict(result=str(Path(job['directory'])/'result.json')))


if __name__ == '__main__':
    main()
