"""A task-independent editing host; generated policies never import this module."""

import importlib
import json
import random
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .model import ModelGateway
from .package import IsolatedPolicy
from .planning import DeterministicContext
from .records import write_json
from .rendering import Renderer
from .scene import certified_oversize, clone_scene, guard, inventory

PROTOCOL = '''Compile the complete user instruction into native EmbodiedGen actions.
Return only JSON with exactly: {"status":"execute|noop|clarify|impossible","delete":["exact_object_id"],"insert":[{"asset":"catalog_key","instance_key":"new_unique_lowercase_id","on_instance":"exact_support_id","in_room":"exact_room_id","place_strategy":"top"}],"message":"brief explanation or clarification question"}.
Only requested actions; preserve every other object. Use the actual current inventory and confirmed history. A missing deletion target is noop, including a repeated deletion. A genuinely ambiguous reference requires clarify, not arbitrary selection. For non-execute statuses both action lists are empty. Use fixed original asset scale and rotation (1.5708,0,0). The native engine determines placement. Do not output code or coordinates. Every new instruction requires interpretation; history is factual evidence, not a new instruction.
'''


def native_manager(path, **options):
    """Physical mesh instances remain editable after a rename containing 'light'.

    The legacy API's default light-fixture filter is unchanged. This opt-in host
    edits visual mesh objects rather than treating their names as light sources.
    """
    api = importlib.import_module(
        'embodied_gen.skills.spatial-computing.api.floorplan_api'
    )
    return api.FloorplanManager(
        str(path), ignore_items=['ceiling', 'exterior'], **options
    )


def validate_plan(plan, names, rooms, catalog):
    if not isinstance(plan, dict) or set(plan) != {
        'status',
        'delete',
        'insert',
        'message',
    }:
        raise ValueError('Expected the four protocol keys')
    if plan['status'] not in [
        'execute',
        'noop',
        'clarify',
        'impossible',
    ] or not isinstance(plan['message'], str):
        raise ValueError('Invalid result status/message')
    if not isinstance(plan['delete'], list) or not isinstance(
        plan['insert'], list
    ):
        raise ValueError('Action fields must be lists')
    if plan['status'] != 'execute' and (plan['delete'] or plan['insert']):
        raise ValueError('Non-execute response cannot modify the scene')
    if plan['status'] == 'execute' and not (plan['delete'] or plan['insert']):
        raise ValueError('Empty execution')
    if len(set(plan['delete'])) != len(plan['delete']) or not set(
        plan['delete']
    ) <= set(names):
        raise ValueError('Deletion is missing, duplicated or protected')
    added = set()
    for item in plan['insert']:
        if set(item) != {
            'asset',
            'instance_key',
            'on_instance',
            'in_room',
            'place_strategy',
        }:
            raise ValueError('Invalid insert schema')
        key = item['instance_key']
        if (
            not isinstance(key, str)
            or not key
            or any(
                c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in key
            )
            or key in set(names) | set(rooms) | added
            or any(k in key for k in ['floor', 'wall'])
        ):
            raise ValueError('Unsafe, protected or duplicate object ID')
        if (
            item['asset'] not in catalog
            or item['on_instance'] not in set(names) - set(plan['delete'])
            or item['in_room'] not in rooms
            or item['place_strategy'] != 'top'
        ):
            raise ValueError('Invalid asset, support, room or placement')
        added.add(key)
    return plan


def same_intent(left, right):
    """A policy may reorder execution, but cannot substitute a parsed target."""
    return (
        left['status'] == right['status']
        and left['delete'] == right['delete']
        and sorted(json.dumps(i, sort_keys=True) for i in left['insert'])
        == sorted(json.dumps(i, sort_keys=True) for i in right['insert'])
    )


class SceneEditor:
    def __init__(
        self,
        catalog,
        policy,
        *,
        gpu_index=4,
        renderer=None,
        planning_mode='legacy',
        lean_parser=False,
        efficient_geometry=False,
    ):
        self.catalog = {k: str(Path(v).resolve()) for k, v in catalog.items()}
        if planning_mode not in ['legacy', 'deterministic']:
            raise ValueError('Unknown planning mode')
        self.policy = Path(policy).resolve() if policy else None
        if planning_mode == 'legacy' and self.policy is None:
            raise ValueError('Legacy planning requires a policy package')
        self.planning_mode = planning_mode
        self.lean_parser = lean_parser
        self.efficient_geometry = efficient_geometry
        self.gpu_index = gpu_index
        self.renderer = renderer

    def _inventory(self, path):
        return inventory(path, geometry_only=self.efficient_geometry)

    def edit(
        self,
        instruction,
        source,
        output_dir,
        *,
        history=None,
        seed=42,
        start_mono=None,
        verify_callback=None,
        video=False,
        max_model_requests=2,
        session_turn=None,
    ):
        """Edit a private copy; return a published scene only after common checks.

        verify_callback is an optional trusted host observer. It is never passed
        to the policy/model and cannot affect what the editor considers legal.
        """
        started = start_mono or time.perf_counter()
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'result.json').exists():
            raise ValueError('Refusing to overwrite an existing edit result')
        source = Path(source).resolve()
        history = history or []
        rec = dict(
            instruction=instruction,
            input=str(source),
            policy=str(self.policy),
            planning_mode=self.planning_mode,
            lean_parser=self.lean_parser,
            efficient_geometry=self.efficient_geometry,
            status='failed',
            output=str(source),
            history=history,
            published=False,
            start_mono=started,
            stages={},
            error=None,
        )
        instructions = (
            Path(__file__).with_name('parser_instructions.txt')
            if self.lean_parser
            else None
        )
        model = ModelGateway(
            directory / 'requests', instructions_file=instructions
        )
        frames = []
        t = time.perf_counter()
        try:
            random.seed(seed)
            np.random.seed(seed)
            if session_turn:
                objects, mapping, facts = session_turn.before
                native = session_turn.manager(source)
            else:
                objects, mapping, facts = self._inventory(source)
                native = native_manager(source)
            names, rooms = (
                sorted(native.get_instance_names()),
                sorted(native.get_room_names()),
            )
            rec['stages']['load_state_s'] = time.perf_counter() - t
            payload = dict(
                instruction=instruction,
                objects=names,
                rooms=rooms,
                assets=sorted(self.catalog),
                history=history,
                facts=facts,
            )
            if session_turn:
                payload['unidentified_objects'] = any(
                    not f['stable_id'] for f in facts
                )
            write_json(directory / 'public_state.json', payload)
            context_builder = (
                DeterministicContext()
                if self.planning_mode == 'deterministic'
                else IsolatedPolicy(self.policy)
            )
            with context_builder as policy:
                t = time.perf_counter()
                prepared = policy.call('prepare_request', payload)
                if set(prepared) - {
                    'context',
                    'guidance',
                    'placement_attempts',
                } or not isinstance(prepared.get('context'), dict):
                    raise ValueError(
                        'Policy must return bounded context/guidance/placement_attempts'
                    )
                attempts = prepared.get('placement_attempts', 1)
                if type(attempts) is not int or attempts not in [1, 2]:
                    raise ValueError(
                        'At most two local placement attempts are available'
                    )
                guidance = prepared.get('guidance', '')
                if (
                    not isinstance(guidance, str)
                    or len(json.dumps(prepared, ensure_ascii=False)) > 24000
                ):
                    raise ValueError('Policy context exceeds its bound')
                # New language and current inventory cannot be removed by policy.
                prompt = (
                    PROTOCOL
                    + '\nPolicy guidance: '
                    + guidance
                    + (
                        '\nCurrent inventory: '
                        + json.dumps(
                            dict(
                                objects=names,
                                rooms=rooms,
                                assets=sorted(self.catalog),
                            ),
                            ensure_ascii=False,
                        )
                        if self.planning_mode == 'legacy'
                        else ''
                    )
                    + '\nContext: '
                    + json.dumps(prepared['context'], ensure_ascii=False)
                    + '\nInstruction: '
                    + instruction
                )
                (directory / 'planner_input.txt').write_text(prompt)
                write_json(directory / 'policy_prepared.json', prepared)
                rec['stages']['prepare_context_s'] = time.perf_counter() - t
                error = None
                for attempt in range(min(2, max_model_requests)):
                    reply = model.query(
                        prompt
                        if error is None
                        else prompt + '\nCorrect this legality error: ' + error
                    )
                    try:
                        plan = validate_plan(
                            json.loads(reply), names, rooms, self.catalog
                        )
                        break
                    except (ValueError, TypeError, KeyError) as exc:
                        error = str(exc)
                        write_json(
                            directory / f'parse_error_{attempt}.json',
                            dict(error=error, reply=reply),
                        )
                        if attempt + 1 >= min(2, max_model_requests):
                            raise
                write_json(directory / 'model_plan.json', plan)
                proposed = policy.call(
                    'decide', dict(plan=plan, facts=facts, history=history)
                )
                if not same_intent(plan, proposed):
                    raise ValueError(
                        'Execution policy changed parsed targets or quantities'
                    )
                plan = validate_plan(proposed, names, rooms, self.catalog)
                rec['plan'] = plan
                write_json(directory / 'executed_plan.json', plan)
                status = plan['status']
                after_path = source
                public_guard = dict(
                    passed=True,
                    added=[],
                    removed=[],
                    preserved=True,
                    geometry=[],
                )
                supports = {}
                checked_objects = None
                if status == 'execute':
                    oversized = [
                        i
                        for i in plan['insert']
                        if certified_oversize(
                            i, self.catalog, objects, mapping
                        )
                    ]
                    if oversized:
                        status = 'impossible'
                        rec['reason'] = (
                            'fixed_orientation_extent_exceeds_support'
                        )
                        rec['certificate'] = [
                            i['instance_key'] for i in oversized
                        ]
                    else:
                        for placement_attempt in range(attempts):
                            t = time.perf_counter()
                            clone = (
                                session_turn.clone
                                if session_turn
                                else clone_scene
                            )
                            path = clone(
                                source,
                                (
                                    session_turn.revision / 'artifact'
                                    if session_turn
                                    else directory
                                )
                                / f'candidate_{placement_attempt}',
                            )
                            manager = (
                                session_turn.manager(path)
                                if session_turn
                                else native_manager(path)
                            )
                            for key in plan['delete']:
                                if not manager.delete_object(key):
                                    raise RuntimeError(
                                        'Native deletion failed: ' + key
                                    )
                            configs = [
                                dict(
                                    asset_path=self.catalog[i['asset']],
                                    instance_key=i['instance_key'],
                                    on_instance=i['on_instance'],
                                    in_room=i['in_room'],
                                    rotation_rpy=[1.5708, 0, 0],
                                    place_strategy='top',
                                    n_max_attempt=2000 // attempts,
                                )
                                for i in plan['insert']
                            ]
                            # Total per-object sampling and search ceilings are unchanged.
                            manager.collector.mesh_sample_num = (
                                10000 // attempts
                            )
                            centers = (
                                manager.batch_insert_objects(configs)
                                if configs
                                else []
                            )
                            write_json(
                                directory / f'native_{placement_attempt}.json',
                                dict(
                                    configs=configs,
                                    centers=centers,
                                    sample_points_per_object=10000 // attempts,
                                    search_attempts_per_object=2000
                                    // attempts,
                                ),
                            )
                            rec['stages'][
                                f'edit_attempt_{placement_attempt}_s'
                            ] = time.perf_counter() - t
                            if not any(center is None for center in centers):
                                candidate = Path(
                                    manager._get_output_path(str(path))
                                )
                                tree = ET.parse(candidate)
                                for link in tree.findall('link'):
                                    if link.get('name') in [
                                        i['instance_key']
                                        for i in plan['insert']
                                    ]:
                                        link.set(
                                            'scene_edit_id', uuid.uuid4().hex
                                        )
                                tree.write(
                                    candidate,
                                    encoding='utf-8',
                                    xml_declaration=True,
                                )
                                if session_turn:
                                    after_objects, _, _ = (
                                        session_turn.inspect_candidate(
                                            candidate, plan, manager
                                        )
                                    )
                                else:
                                    after_objects, _, _ = self._inventory(
                                        candidate
                                    )
                                public_guard, supports = guard(
                                    objects, after_objects, plan, mapping
                                )
                                if public_guard['passed']:
                                    after_path = candidate
                                    checked_objects = after_objects
                                    break
                                status = 'failed'
                                rec['reason'] = (
                                    'public_geometry_or_preservation_check'
                                )
                            else:
                                status = 'failed'
                                rec['reason'] = 'native_search_exhausted'
                            raw = dict(
                                plan=plan,
                                error=rec['reason'],
                                attempt=placement_attempt + 1,
                            )
                            write_json(
                                directory
                                / f'raw_feedback_{placement_attempt}.json',
                                raw,
                            )
                            feedback = policy.call('process_tool_result', raw)
                            write_json(
                                directory
                                / f'delivered_feedback_{placement_attempt}.json',
                                feedback,
                            )
                            if (
                                not feedback.get('retry')
                                or placement_attempt + 1 >= attempts
                            ):
                                break
                            retry_plan = feedback.get('plan', plan)
                            if not same_intent(plan, retry_plan):
                                raise ValueError(
                                    'Recovery changed the instruction constraints'
                                )
                            plan = retry_plan
                            status = 'execute'
                        if status == 'execute' and after_path == source:
                            status = 'failed'
                if status != 'execute':
                    after_path = source
                    public_guard = dict(
                        passed=True,
                        added=[],
                        removed=[],
                        preserved=True,
                        geometry=[],
                    )
                    supports = {}
                t = time.perf_counter()
                manager = (
                    session_turn.manager(after_path)
                    if session_turn
                    else native_manager(after_path)
                )
                manager.visualize(str(directory / 'floorplan.png'))
                rec['stages']['floorplan_s'] = time.perf_counter() - t
                rec['edit_end_mono'] = time.perf_counter()
                t = time.perf_counter()
                if self.renderer is None:
                    self.renderer = Renderer(self.gpu_index)
                self.renderer.reset()
                before_load = self.renderer.preview(
                    source, directory, 'before'
                )
                after_load = self.renderer.preview(
                    after_path, directory, 'after'
                )
                rec['gpu'] = self.renderer.gpu()
                write_json(directory / 'gpu.json', rec['gpu'])
                if self.efficient_geometry:
                    after_objects = (
                        checked_objects if status == 'execute' else objects
                    )
                    if after_objects is None:
                        raise RuntimeError('Missing verified scene state')
                else:
                    after_objects, _, _ = self._inventory(after_path)
                release = {}
                if public_guard['added']:
                    release, frames = self.renderer.release(
                        after_objects,
                        public_guard['added'],
                        directory,
                        supports,
                        video=video,
                    )
                rec['stages']['verification_s'] = time.perf_counter() - t
                common_ok = public_guard['passed'] and all(
                    r['stable'] for r in release.values()
                )
                if not common_ok:
                    status = 'failed'
                    after_path = source
                    rec['reason'] = 'support_diagnostic_failed'
                rec.update(
                    status=status,
                    common_ok=bool(common_ok),
                    guard=public_guard,
                    release=release,
                    before_load=before_load,
                    after_load=after_load,
                )
                if verify_callback:
                    rec['evaluation'] = verify_callback(
                        source, after_path, status, rec
                    )
                if common_ok and status in [
                    'execute',
                    'noop',
                    'clarify',
                    'impossible',
                ]:
                    # Publication is a path to a fully materialized scene, not a partial write.
                    rec['history'] = history + [
                        dict(
                            instruction=instruction,
                            status=status,
                            deleted=plan['delete']
                            if status == 'execute'
                            else [],
                            inserted=[
                                dict(
                                    i,
                                    stable_id=after_objects[i['instance_key']][
                                        'link'
                                    ].get('scene_edit_id'),
                                )
                                for i in plan['insert']
                            ]
                            if status == 'execute'
                            else [],
                        )
                    ]
                    if session_turn:
                        session_turn.publish(
                            after_path, rec['history'], plan, status
                        )
                        rec['session'] = session_turn.metrics
                    rec.update(output=str(after_path), published=True)
                    write_json(
                        directory / 'published.json',
                        dict(urdf=str(after_path), status=status),
                        once=True,
                    )
                rec['policy_calls'] = policy.calls
                write_json(directory / 'policy_calls.json', policy.calls)
        except Exception as exc:
            if session_turn and not rec['published']:
                rec.update(
                    status='failed', history=history, output=str(source)
                )
            rec['error'] = repr(exc)
            (directory / 'error.txt').write_text(traceback.format_exc())
        rec['verified_mono'] = time.perf_counter()
        rec['verified_s'] = rec['verified_mono'] - started
        rec['edit_s'] = (
            rec.get('edit_end_mono', rec['verified_mono']) - started
        )
        rec['requests'] = model.records
        rec['known_tokens'] = sum(
            (r.get('usage') or {}).get('input_tokens', 0)
            + (r.get('usage') or {}).get('output_tokens', 0)
            for r in model.records
        )
        rec['unknown_usage_requests'] = sum(
            r.get('usage') is None for r in model.records
        )
        rec['llm_rpc_s'] = sum(r['latency_s'] for r in model.records)
        rec['non_llm_s'] = rec['verified_s'] - rec['llm_rpc_s']
        if frames:
            t = time.perf_counter()
            import imageio.v2 as imageio

            imageio.mimwrite(
                directory / 'release.mp4', frames, fps=30, macro_block_size=1
            )
            rec['video_encoding_s'] = time.perf_counter() - t
        write_json(directory / 'result.json', rec)
        return rec
