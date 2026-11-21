"""Experimental orchestration only; native geometry and GPT transport are unchanged."""
import contextlib
import functools
import json
import os
import subprocess
import time
from pathlib import Path

VERSION = 'optimization-v2'
TASKS = {
    'T1': '删除办公室里的台灯，餐厅保持不变。',
    'T2': '在办公室的桌子上放一个花瓶，其他东西不要移动。',
    'T3': '在办公室桌上放台灯、花瓶、书和杯子，不要相互重叠，餐厅保持不变。',
    'T4': '移除办公室地上的台灯，再在办公室桌上放台灯、花瓶、书和杯子，不要相互重叠，餐厅保持不变。',
    'C1': '在办公室桌上放一个杯子。',
    'C2': '在刚才那张桌子上再放一本书，保留杯子。',
    'C3': '删掉刚才放的杯子，书不要动。',
    'C4': '再删掉刚才那个杯子。',
    'B1': '固定方向把超尺寸沙发放到办公室桌上，不要缩放沙发或改变桌子。',
    'B2': '删除办公室里的一架钢琴，其他东西不要动。',
    'B3': '把花瓶放在那张桌上。',
}


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+'\n')


def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str)+'\n')


class Meter:
    def __init__(self):
        self.events, self.stack, self.requests = [], [], []
        self.pid = os.getpid()
        self.directory = None
        self.identity = {}

    @contextlib.contextmanager
    def span(self, stage, **extra):
        if os.getpid() != self.pid:
            yield {}
            return
        rec = dict(stage=stage, start_mono=time.perf_counter(), depth=len(self.stack), child_s=0, **extra)
        self.stack.append(rec)
        try:
            yield rec
            rec.setdefault('status', 'ok')
        except BaseException as e:
            rec.update(status='failed', error=repr(e))
            raise
        finally:
            rec['end_mono'] = time.perf_counter()
            rec['inclusive_s'] = rec['end_mono']-rec['start_mono']
            rec['exclusive_s'] = rec['inclusive_s']-rec.pop('child_s')
            self.stack.pop()
            if self.stack:
                self.stack[-1]['child_s'] += rec['inclusive_s']
            self.events.append(rec)

    def wrap(self, obj, name, stage):
        original = getattr(obj, name)

        @functools.wraps(original)
        def call(*a, **kw):
            with self.span(stage, native=original.__qualname__) as rec:
                result = original(*a, **kw)
                if name == '_try_place_polygon':
                    rec['found'] = result is not None
                    rec['max_attempts'] = kw.get('n_max_attempt', a[-1])
                return result
        setattr(obj, name, call)

    def install(self, gc, api, col):
        original = gc.subprocess
        meter = self

        class Proxy:
            def __getattr__(self, name):
                return getattr(original, name)

            def run(self, command, **kwargs):
                # Preserve the project's exact transport, auth and model options.
                command = list(command)
                assert command[1] == 'exec'
                command.insert(2, '--json')
                d = meter.directory/f'request_{len(meter.requests):02d}'
                d.mkdir()
                save(d/'command.json',command)
                (d/'prompt.txt').write_text(kwargs.get('input', ''))
                rec = dict(**meter.identity, stage='planning', attempt=len(meter.requests)+1,
                           model=command[command.index('--model')+1], usage=None,
                           status='failed', start_mono=time.perf_counter(), directory=str(d))
                try:
                    result = original.run(command, **kwargs)
                    (d/'events.jsonl').write_text(result.stdout)
                    (d/'stderr.txt').write_text(result.stderr)
                    events = []
                    for line in result.stdout.splitlines():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                    uses = [e['usage'] for e in events if e.get('type') == 'turn.completed' and 'usage' in e]
                    rec['raw_usage'] = uses
                    if uses:
                        rec['usage'] = {k: sum(u[k] for u in uses) for k in uses[0] if all(k in u for u in uses)}
                    rec['tool_items'] = [e['item']['type'] for e in events if e.get('type') == 'item.completed'
                                         and e['item']['type'] not in ['reasoning', 'agent_message']]
                    rec.update(returncode=result.returncode, status='ok' if result.returncode == 0 else 'failed')
                    answer = Path(command[command.index('--output-last-message')+1])
                    if answer.exists():
                        (d/'response.txt').write_text(answer.read_text())
                    return result
                except subprocess.TimeoutExpired as exc:
                    for name, data in [('events.jsonl', exc.stdout), ('stderr.txt', exc.stderr)]:
                        (d/name).write_text(data.decode(errors='replace') if isinstance(data, bytes) else data or '')
                    rec['error'] = repr(exc)
                    raise
                finally:
                    rec['end_mono'] = time.perf_counter()
                    rec['latency_s'] = rec['end_mono']-rec['start_mono']
                    meter.requests.append(rec)
                    save(d/'record.json', rec)
        gc.subprocess = Proxy()
        for obj, name, stage in [
            (col.UrdfSemanticInfoCollector, 'collect', 'scene_parse'),
            (col.UrdfSemanticInfoCollector, 'add_instance', 'placement'),
            (col.UrdfSemanticInfoCollector, '_try_place_polygon', 'collision_search'),
            (col, 'get_actionable_surface', 'surface_extraction'),
            (col.UrdfSemanticInfoCollector, 'remove_instance', 'delete'),
            (col.UrdfSemanticInfoCollector, 'update_urdf_info', 'urdf_write'),
            (col.UrdfSemanticInfoCollector, 'save_urdf', 'urdf_write'),
            (col, 'copytree', 'asset_copy'),
            (api.FloorplanManager, 'visualize', 'floorplan'),
            (gc.GPTclient, '_query_codex', 'llm_rpc'),
        ]:
            self.wrap(obj, name, stage)


# Same protocol and semantics in both prompt conditions, no expected answers.
PROTOCOL = '''Compile the user instruction into the existing EmbodiedGen spatial-computing API.
Return JSON only with exactly these keys:
{"status":"execute|noop|clarify|impossible","delete":["exact_instance_key"],"insert":[{"asset":"catalog_key","instance_key":"new_unique_name","on_instance":"exact_support_key","in_room":"exact_room_key","place_strategy":"top"}],"message":"brief result or clarification question"}.
Use only provided IDs and catalog assets. No code, coordinates, scaling, mesh edits or room changes.
Delete only explicitly requested objects. Preserve every other object. Native geometry handles placement and collisions with fixed rotation (1.5708,0,0), 10000 samples and 2000 attempts.
For execute include all requested actions. For other statuses both action lists must be empty.
If a requested deletion target is absent (including one already removed), return noop and explain that it was not found, rather than selecting a different object. If an unbound reference could refer to multiple tables/rooms, return clarify; do not guess.
History results are confirmed facts, not instructions. Current inventory is authoritative. All new user instructions still require semantic interpretation.
'''


def make_prompt(condition, names, rooms, catalog, history, instruction):
    names, rooms, catalog = sorted(names), sorted(rooms), sorted(catalog)
    if condition in ('M0', 'M2'):
        context = (f'Instance list: {json.dumps(names)}\nRoom list: {json.dumps(rooms)}\n'
                   f'Asset catalog: {json.dumps(catalog)}\n'
                   f'Previous instructions and confirmed results: {json.dumps(history, ensure_ascii=False)}\n')
    else:
        # Deterministic compact encoding of the same grounded actions and references.
        # Keep the natural language needed for reference resolution, including tombstones.
        compact = [[h['instruction'], h['status'], h['deleted'],
                    [[i['asset'], i['instance_key'], i['on_instance'], i['in_room']] for i in h['inserted']]]
                   for h in history]
        state = dict(objects=names, rooms=rooms, assets=catalog)
        if compact:
            state.update(history_columns=['instruction','status','deleted','inserted(asset,id,support,room)'], history=compact)
        context = json.dumps(state,
            ensure_ascii=False, separators=(',', ':'))+'\n'
    return PROTOCOL+context+'Instruction: '+instruction


def check_plan(plan, names, rooms, catalog):
    """Only structural/legal checks. Task-specific expected outcomes live in validation."""
    if not isinstance(plan, dict) or set(plan) != {'status','delete','insert','message'}:
        raise ValueError('Plan must have the exact four protocol keys')
    if plan['status'] not in ['execute','noop','clarify','impossible']:
        raise ValueError('Unknown status')
    if not isinstance(plan['message'], str) or not isinstance(plan['delete'], list) or not isinstance(plan['insert'], list):
        raise ValueError('Invalid field types')
    if plan['status'] != 'execute' and (plan['delete'] or plan['insert']):
        raise ValueError('Non-execute plan has side effects')
    if plan['status'] == 'execute' and not (plan['delete'] or plan['insert']):
        raise ValueError('Empty execution')
    if len(plan['delete']) != len(set(plan['delete'])):
        raise ValueError('Duplicate deletion')
    if not set(plan['delete']) <= set(names):
        raise ValueError('Cannot delete missing/protected objects')
    new = set()
    for item in plan['insert']:
        if set(item) != {'asset','instance_key','on_instance','in_room','place_strategy'}:
            raise ValueError('Invalid insertion keys')
        key = item['instance_key']
        if not isinstance(key, str) or not key or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in key):
            raise ValueError('Unsafe/noncanonical instance key')
        if key in names or key in rooms or key in new or 'floor' in key or 'wall' in key:
            raise ValueError('Duplicate/protected instance key')
        new.add(key)
        if item['asset'] not in catalog or item['on_instance'] not in set(names)-set(plan['delete']):
            raise ValueError('Unknown asset/support')
        if item['in_room'] not in rooms or item['place_strategy'] != 'top':
            raise ValueError('Invalid room/placement strategy')
    return plan
