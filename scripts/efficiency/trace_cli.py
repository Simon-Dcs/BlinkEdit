"""Meter the unmodified spatial-computing CLI. No placement/planning replacement.

Only --json is added to native Codex requests. Inclusive/exclusive monotonic
timers wrap native calls, with raw prompts, responses and usage retained.
"""
import argparse
import functools
import importlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

START = time.perf_counter()
START_ISO = datetime.now().astimezone().isoformat()
p = argparse.ArgumentParser()
p.add_argument('--run-dir', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--plan-request', help='Optional separately-labelled outer planner experiment')
p.add_argument('--asset-root', help='Official downloaded asset catalog for outer planner')
args, native_args = p.parse_known_args()
out = Path(args.run_dir)
out.mkdir(parents=True, exist_ok=True)
events, requests, stack = [], [], []
main_pid = os.getpid()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str)+'\n')


def wrap(obj, name, stage):
    original = getattr(obj, name)

    @functools.wraps(original)
    def call(*a, **kw):
        if os.getpid() != main_pid:
            return original(*a, **kw)
        rec = dict(stage=stage, native=f'{original.__module__}.{original.__qualname__}',
                   offset_s=time.perf_counter()-START, depth=len(stack), child_s=0)
        t = time.perf_counter()
        stack.append(rec)
        try:
            result = original(*a, **kw)
            rec['status'] = 'ok'
            if name.startswith('resolve_') or name in ['add_instance', 'batch_insert_objects', 'insert_object', 'delete_object']:
                rec['result'] = result
            if name == '_try_place_polygon':
                rec['found_placement'] = result is not None
                rec['max_attempts'] = kw.get('n_max_attempt', a[-1] if a else None)
            if name == 'collect':
                rec['instances'] = list(a[0].instances)
            return result
        except BaseException as exc:
            rec.update(status='failed', error=str(exc))
            raise
        finally:
            rec['inclusive_s'] = time.perf_counter()-t
            rec['exclusive_s'] = rec['inclusive_s']-rec.pop('child_s')
            stack.pop()
            if stack:
                stack[-1]['child_s'] += rec['inclusive_s']
            events.append(rec)
    setattr(obj, name, call)


def install_meter(gc):
    original = gc.subprocess

    class Meter:
        def __getattr__(self, key):
            return getattr(original, key)

        def run(self, command, **kwargs):
            command = list(command)
            assert command[1] == 'exec'
            command.insert(2, '--json')
            d = out/f'request_{len(requests):02d}'
            d.mkdir()
            (d/'prompt.txt').write_text(kwargs.get('input', ''))
            rec = dict(started_at=datetime.now().astimezone().isoformat(),
                       model=command[command.index('--model')+1] if '--model' in command else 'CLI default',
                       reasoning_effort='medium', provider='codex', authentication='ChatGPT',
                       usage=None, status='failed', directory=str(d))
            t = time.perf_counter()
            try:
                result = original.run(command, **kwargs)
                (d/'events.jsonl').write_text(result.stdout)
                (d/'stderr.txt').write_text(result.stderr)
                parsed = []
                for line in result.stdout.splitlines():
                    try:
                        parsed.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                usage = [x['usage'] for x in parsed if x.get('type') == 'turn.completed' and 'usage' in x]
                rec['usage_events'] = usage
                if usage:
                    rec['usage'] = {k: sum(u[k] for u in usage) for k in usage[0] if all(k in u for u in usage)}
                rec['tool_items'] = [x['item']['type'] for x in parsed if x.get('type') == 'item.completed' and x['item']['type'] not in ['reasoning', 'agent_message']]
                rec.update(returncode=result.returncode, status='ok' if result.returncode == 0 else 'failed')
                answer = Path(command[command.index('--output-last-message')+1])
                if answer.exists():
                    (d/'response.txt').write_text(answer.read_text())
                return result
            except subprocess.TimeoutExpired as exc:
                for fn, data in [('events.jsonl', exc.stdout), ('stderr.txt', exc.stderr)]:
                    (d/fn).write_text(data.decode(errors='replace') if isinstance(data, bytes) else data or '')
                rec['error'] = str(exc)
                raise
            finally:
                rec['seconds'] = time.perf_counter()-t
                requests.append(rec)
                save(d/'record.json', rec)
    gc.subprocess = Meter()


code = 0
try:
    t = time.perf_counter()
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    cli = importlib.import_module('embodied_gen.skills.spatial-computing.cli.main')
    api = importlib.import_module('embodied_gen.skills.spatial-computing.api.floorplan_api')
    col = importlib.import_module('embodied_gen.skills.spatial-computing.core.collector')
    gc = importlib.import_module('embodied_gen.utils.gpt_clients')
    events.append(dict(stage='imports', inclusive_s=time.perf_counter()-t, exclusive_s=time.perf_counter()-t, depth=0))
    install_meter(gc)
    for name in ['resolve_on_instance', 'resolve_in_room', 'resolve_beside_instance', 'resolve_delete_instance']:
        wrap(api.FloorplanManager, name, 'semantic_resolution')
    for obj, name, stage in [
        (col.UrdfSemanticInfoCollector, 'collect', 'scene_parse'),
        (col.UrdfSemanticInfoCollector, 'add_instance', 'placement'),
        (col, 'get_actionable_surface', 'surface_extraction'),
        (col.UrdfSemanticInfoCollector, '_try_place_polygon', 'collision_search'),
        (col.UrdfSemanticInfoCollector, 'remove_instance', 'delete'),
        (col.UrdfSemanticInfoCollector, 'update_urdf_info', 'urdf_write'),
        (col.UrdfSemanticInfoCollector, 'save_urdf', 'urdf_write'),
        (col, 'copytree', 'asset_copy'),
        (api.FloorplanManager, 'visualize', 'floorplan'),
        (gc.GPTclient, '_query_codex', 'llm_rpc'),
    ]:
        wrap(obj, name, stage)
    cfg = cli.tyro.cli(cli.FloorplanConfig, args=native_args)
    save(out/'config.json', vars(cfg))
    if args.plan_request:
        # Outer orchestration experiment: GPT only produces a bounded action
        # description. All spatial decisions, collisions and edits stay native.
        manager = api.FloorplanManager(urdf_path=cfg.urdf_path)
        catalog = {a: str(Path(args.asset_root)/a/'mesh'/f'{a}.obj') for a in ['lamp','vase','book','mug']}
        prompt = f'''Compile the user instruction into the existing EmbodiedGen spatial-computing API.
Return JSON only: {{"delete": ["exact_instance_key"], "insert": [{{"asset": "catalog_key", "instance_key": "new_unique_name", "on_instance": "exact_support_key", "in_room": "exact_room_key", "place_strategy": "top"}}]}}.
Use the exact identifiers provided. Do not output code, coordinates or mesh edits.
Delete only explicitly requested instances. Insert only requested catalog assets.
Leave other instances untouched. The native engine handles collision-aware placement.
Instance list: {json.dumps(manager.get_instance_names())}
Room list: {json.dumps(manager.get_room_names())}
Asset catalog: {json.dumps(list(catalog))}
Instruction: {args.plan_request}'''
        reply = gc.GPT_CLIENT.query(prompt)
        if not reply:
            raise RuntimeError('Outer planner returned no response')
        plan = json.loads(reply)
        save(out/'plan.json', plan)
        assert set(plan) == {'delete', 'insert'}
        names = set(manager.get_instance_names())
        rooms = set(manager.get_room_names())
        for key in plan['delete']:
            assert key in names and key not in rooms
            if not manager.delete_object(key):
                raise RuntimeError(f'Deletion failed: {key}')
        objects = []
        for item in plan['insert']:
            assert set(item) == {'asset','instance_key','on_instance','in_room','place_strategy'}
            assert item['asset'] in catalog
            assert item['on_instance'] in names-set(plan['delete'])
            assert item['in_room'] in rooms
            assert item['instance_key'] not in names
            assert item['place_strategy'] == 'top'
            objects.append(dict(asset_path=catalog[item['asset']], instance_key=item['instance_key'],
                                on_instance=item['on_instance'], in_room=item['in_room'],
                                rotation_rpy=[1.5708,0,0],place_strategy='top'))
        centers = manager.batch_insert_objects(objects)
        if any(center is None for center in centers):
            raise RuntimeError('Native batch placement failed')
        manager.visualize(output_path=cfg.output_path)
    else:
        cli.entrypoint(cfg)
except BaseException as exc:
    import traceback
    code = exc.code if isinstance(exc, SystemExit) else 1
    if code:
        traceback.print_exc()
finally:
    save(out/'trace.json', dict(started_at=START_ISO, seconds=time.perf_counter()-START,
                               seed=args.seed, exitcode=code, events=events, requests=requests,
                               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
sys.exit(code)
