"""One cold process per benchmark turn. No scoring data is sent to a model tool."""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from agents import ask, regular
from sandbox import isolated
from embodied_gen.harness.asset_edit import gltf, program
from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.state_store import (
    RevisionStore,
    read,
    stamps,
)

CLOCK = time.perf_counter
GLB_PROMPT = """Modify the current segmented 3D object to follow this instruction and the supplied target image. Keep other parts. Construct new geometry in code. Return only Python code assigning a list of actions to ops. Available: Python standard library and state (the inventory below). The host executes these native glTF operations:
{"op":"remove","id":integer node key} removes just its geometry, not its children.
{"op":"paint","id":integer,"color":[r,g,b],"roughness":0.6,"metallic":0} creates a new material for this node, keeping its UVs; RGB is linear 0..1. Optional keep_texture=true to tint the existing base texture, or pixels=H x W x 3 array of RGB bytes (generate patterns in Python).
{"op":"add","shape":"box|sphere|cylinder|cone","name":"...","center":[x,y,z],"size":[sx,sy,sz],"rotation":[rx,ry,rz],"color":[r,g,b]}: size=full dimensions, XYZ angles in degrees, default axis +Y. Optional attach_to: existing parent node key; coordinates remain world-space. Use multiple shapes or custom triangles {"op":"add","shape":"mesh","name":"...","vertices":[[x,y,z],...],"faces":[[i,j,k],...],"color":[r,g,b]}. Replace by remove and add. Use glTF world axes (+Y up), NOT Blender axes. Outputs become the next turn's actual object. No external files or libraries required.
"""


def render_program(root, original, edited, out, tools, ce, measure_levels=False):
    from evaluation.prompts import VIEWS

    out.mkdir()
    with tempfile.TemporaryDirectory() as w:
        w = Path(w)
        shutil.copyfile(original, w / 'source.py')
        shutil.copyfile(edited, w / 'modified.py')
        task = dict(
            original_script='/work/source.py',
            modified_script='/work/modified.py',
            output_dir='/work',
            views=VIEWS,
            render={'samples': 64, 'resolution': 512},
            target_coverage=0.95,
            python_paths=['/deps'],
        )
        write_json(w / 'task.json', task)
        status = isolated(
            [
                '/blender/blender',
                '-b',
                '--threads',
                os.environ.get('BENCH_RENDER_THREADS', '1'),
                '--python-exit-code',
                '1',
                '--python',
                '/observe.py' if measure_levels else '/render.py',
                '--',
                '--task',
                '/work/task.json',
            ],
            w,
            readonly=[
                (tools / 'blender-5.1.0-linux-x64', '/blender'),
                (tools / 'blender_deps', '/deps'),
                (ce / 'evaluation/blender_render.py', '/render.py'),
                *([(Path(__file__).parent/'study3/measure_render.py', '/observe.py')] if measure_levels else []),
            ],
            timeout=int(os.environ.get('BENCH_RENDER_TIMEOUT', '480')),
            log=out / 'execution.log',
        )
        if status:
            return False
        for name in ['frame.json'] + [
            f'{s}_{v}.png' for v in VIEWS for s in ('original', 'modified')
        ]:
            shutil.copyfile(regular(w / name), out / name)
        if measure_levels:
            shutil.copyfile(regular(w/'levels.json'),out/'levels.json')
        return read(out / 'frame.json')['modified_has_geometry']


def review(out, metrics, instruction, valid, err):
    from evaluation.prompts import VIEWS
    if not valid:
        metrics.update(
            accept=False, feedback='Blender execution failed: ' + err
        )
    else:
        pics = [
            out / 'views' / f'{s}_{v}.png'
            for v in VIEWS
            for s in ('original', 'modified')
        ]
        prompt = (
            'You are a separate visual reviewer. Given the original instruction and six pairs of renders '
            '(original then modified in order front, back, left, right, top, bottom), check each requested clause '
            'and preservation using visible evidence. Give specific repair advice for failed clauses. '
            'Return JSON {"accept":true or false,"feedback":"per-clause findings and advice"}.\n'
            + instruction
        )
        reply = ask(out / 'reviewer', prompt, pics, stage='reviewer')
        result = json.loads(
            reply.strip()
            .removeprefix('```json')
            .removesuffix('```')
            .strip()
        )
        if type(result['accept']) is not bool:
            raise ValueError('Invalid reviewer verdict')
        metrics.update(
            accept=result['accept'], feedback=result['feedback']
        )


def run(root, kind, key, arm, turn, stage, start):
    from evaluation.extract import extract_program
    from evaluation.prompts import VIEWS, editor_messages

    manifest = read(root / 'manifest.json')
    ce = Path(manifest['roots']['code'])
    tools = root.parent / 'tools'
    data = root / 'setup/data'
    group = root / kind / stage / key / arm
    out = group / f'turn{turn:02}'
    out.mkdir(parents=True, exist_ok=False)
    metrics = dict(
        kind=kind,
        case=key,
        arm=arm,
        turn=turn,
        completed=False,
        start=time.time() - (CLOCK() - start),
    )
    method = manifest.get('method_roles', {}).get(arm, arm)
    ruled = method in ('F0R', 'F1R', 'F2', 'F2T')
    if ruled:
        from study import rules
    store = RevisionStore(group / 'session') if method in ('F1', 'F1R') else None
    try:
        with store.writer() if store else contextlib.nullcontext():
            lookup_start = CLOCK()
            prior = store.load() if store else None
            last = (
                read(group / f'turn{turn - 1:02}/outcome.json')
                if turn > 1
                else {}
            )
            if kind == 'edithero':
                original = data / 'states' / key / 'turn00.glb'
                instruction = manifest['chain_info'][key]['instructions'][
                    turn - 1
                ]
            else:
                # Keep eval_query/checklist out of the editor adapter and the workspace.
                pub = read(root / 'setup/public_code' / f'{key}.json')
                original = ce / 'data/sources' / f"{pub['source_id']}.py"
                instruction = pub['instruction']
            source = (
                prior['artifact']
                if prior
                else Path(last.get('artifact', original))
            )
            expected = prior['dependencies'] if prior else stamps([source])
            rev = store.begin() if store else None
            t = CLOCK()
            schema = (
                'embedded-glb-v2'
                if kind == 'edithero'
                else 'blender-program-v1'
            )
            if (
                prior
                and prior['state'] is not None
                and prior['state'].get('schema') == schema
            ):
                before = prior['state']
                metrics['cached'] = True
            elif kind == 'edithero':
                before, _ = gltf.observe(gltf.header(source)[0])
                metrics['cached'] = False
            else:
                before = program.observe(source.read_text())
                metrics['cached'] = False
            metrics['observe_s'] = CLOCK() - t
            metrics['state_setup_s'] = CLOCK() - lookup_start
            history = prior['history'] if prior else last.get('history', [])
            dest = (rev / 'artifact' if rev else out) / (
                'edited.glb' if kind == 'edithero' else 'edited.py'
            )
            if kind == 'edithero':
                public = dict(
                    before,
                    nodes={
                        k: {
                            key: val
                            for key, val in n.items()
                            if key != 'stable_id'
                        }
                        for k, n in before['nodes'].items()
                    },
                )
                prompt = (
                    GLB_PROMPT
                    + '\nCurrent state: '
                    + json.dumps(public)
                    + '\nPrevious instructions: '
                    + json.dumps(history)
                    + '\nNew instruction: '
                    + instruction
                )
                imgs = [
                    data
                    / 'renders'
                    / key
                    / f'turn{turn:02}'
                    / 'view0_cond.png'
                ]
                direct = {'source.glb': source}
            else:
                system, user = editor_messages(
                    {'instruction': instruction}, original.read_text()
                )
                prompt = system + '\n' + user
                if method != 'direct':
                    prompt += (
                        '\nProgram index (static analysis, not visual information): '
                        + json.dumps(before)
                    )
                if turn > 1:
                    prompt += (
                        '\nCurrent full program:\n'
                        + Path(last['attempted']).read_text()
                        + '\nFeedback: '
                        + last['feedback']
                    )
                imgs = []
                direct = {'source.py': original, 'current.py': source}
            local_ops = local_program = None
            archive = last.get('retired_parts', {}) if ruled else {}
            if ruled:
                if kind == 'edithero':
                    local_ops = rules.eh(instruction, before, source, archive, last.get('created_parts', []))
                elif turn == 1:
                    local_program = rules.ce(instruction, source.read_text())
            metrics['route'] = 'local' if local_ops or local_program else 'model'
            selected_id = None
            if method in ('F2', 'F2T') and not (local_ops or local_program):
                if kind == 'edithero':
                    from embodied_gen.harness.asset_edit import grounding
                    proposal = grounding.propose(instruction, before, source, out)
                    if proposal:
                        how, value, evidence = proposal
                        if how == 'measured':
                            selected_id = value
                        else:
                            question = ('Select a SINGLE leaf mesh to remove from the CURRENT asset to satisfy the instruction. '
                                        'First image = numbered selection candidates in the current asset, with individual meshes below; '
                                        'second image = desired result. Compare visible identity and position. '
                                        'Return JSON {"id":integer} or {"id":null} if ambiguous, multiple meshes are needed, '
                                        'or the target is absent. Instruction: ' + instruction +
                                        '\nCandidates: ' + json.dumps({k:before['nodes'][k]['name'] for k in evidence['ids']}))
                            reply = ask(out/'locator', question, [value,*imgs])
                            selected_id = grounding.selected(reply, evidence['ids'])
                        evidence.update(method=how,selected=selected_id)
                        write_json(out/'grounding.json', evidence)
                        if selected_id is not None:
                            local_ops=[{'op':'remove','id':selected_id}]
                            metrics['route']='measured_remove' if how=='measured' else 'model_select'
                else:
                    from embodied_gen.harness.asset_edit import local_program as patching
                    current = Path(last['attempted']).read_text() if turn>1 else source.read_text()
                    feedback = last.get('feedback','')
                    measured = last.get('unique_level_surfaces')
                    if measured:
                        feedback += '\nWorld-Z measurements of uniquely named current shelf surfaces (before camera normalization): '+json.dumps(measured)
                    evidence_note = ''
                    if method == 'F2T':
                        from study4.trace_call import locate
                        spans, binding = locate(root, out, current, instruction, feedback)
                        if spans:
                            evidence_note = ('\nObserved in the execution of the CURRENT program: these visible meshes were created '
                                             'at the listed source lines. Bounds are world coordinates, not necessarily the design axes. '
                                             'Preserve attached surfaces and unrelated geometry.\n' + json.dumps(binding['proof']))
                    else:
                        spans = patching.windows(current, instruction, feedback)
                    if spans:
                        reply=ask(out/'patcher',patching.patch_prompt(current,spans,instruction,feedback)+evidence_note)
                        merged=patching.apply_reply(current,spans,reply)
                        write_json(out/'patch.json',dict(spans=spans,merged=merged is not None))
                        if merged is not None:
                            local_program=(merged,'patch')
                            metrics['route']='model_patch'
                        else:
                            local_program=None
            if local_ops:
                code = 'ops=' + repr(local_ops)
            elif local_program:
                code = local_program[0]
            else:
                actor = ask(out/'editor', prompt, imgs, direct_files=direct if method=='direct' else None)
                code = extract_program(actor)
            t = CLOCK()
            err = ''
            if kind == 'edithero':
                with tempfile.TemporaryDirectory() as wd:
                    w = Path(wd)
                    (w / 'answer.py').write_text(code)
                    write_json(w / 'state.json', public)
                    (w / 'run.py').write_text(
                        "import json,runpy\nstate=json.load(open('state.json'))\nx=runpy.run_path('answer.py',init_globals={'state':state});json.dump(x['ops'],open('actions.json','w'),allow_nan=False)"
                    )
                    rc = isolated(
                        ['/usr/bin/python3', '-I', 'run.py'],
                        w,
                        log=out / 'execution.log',
                        timeout=30,
                    )
                    if rc:
                        raise ValueError('Action program failed')
                    ops = read(regular(w / 'actions.json'))
                write_json(out / 'actions.json', ops)
                if ruled:
                    archive = rules.removed_doc(gltf.header(source)[0], ops, before, archive)
                    metrics['retired_parts'] = archive
                after, n_calc = gltf.edit(
                    source,
                    dest,
                    ops,
                    before,
                    store is not None,
                    initialize_ids=turn == 1,
                    trusted_restores={int(o['id']):o['mesh'] for o in (local_ops or []) if o['op']=='restore'},
                )
                if selected_id is not None:
                    grounding.verify_removed(before, after, selected_id)
                if ruled:
                    metrics['created_parts'] = rules.additions_doc(instruction, ops, before, after, last.get('created_parts', []))
                metrics['artifact_s'] = CLOCK() - start
                # Interpret the delivered active scene independently, not the number of requested actions.
                import trimesh

                scene = trimesh.load(dest, force='scene', skip_materials=True)
                if not scene.graph.nodes_geometry:
                    raise ValueError('Unusable GLB')
                metrics['nodes_recomputed'] = n_calc
                history = history + [instruction]
                retired_ids = [
                    gltf.header(dest)[0]['nodes'][int(o['id'])]
                    .get('extras', {})
                    .get('scene_edit_id')
                    for o in ops
                    if o['op'] == 'remove'
                ]
                delta = dict(
                    instruction=instruction, actions=ops, retired=retired_ids
                )
                valid = True
            else:
                dest.write_text(code)
                metrics['artifact_s'] = CLOCK() - start
                metrics['attempted'] = str(dest)
                try:
                    after = program.observe(code)
                    valid = render_program(
                        root, original, dest, out / 'views', tools, ce,
                        measure_levels=(method in ('F2', 'F2T') and all(w in instruction.lower() for w in ('front','middle')) and any(w in instruction.lower() for w in ('rear','back'))),
                    )
                except (SyntaxError, subprocess.TimeoutExpired) as exc:
                    err = str(exc)
                    valid = False
                if (out/'views/levels.json').exists():
                    levels=read(out/'views/levels.json')
                    if 'unique_surfaces' in levels: metrics['unique_level_surfaces']=levels['unique_surfaces']
                if not valid and not err:
                    err = (out / 'views/execution.log').read_text(
                        errors='replace'
                    )[-1800:]
                history = history + [instruction]
                delta = dict(
                    instruction=instruction, full_program_rebuilt=True
                )
            metrics['execute_s'] = CLOCK() - t
            if valid and store:
                store.commit(
                    rev,
                    dest,
                    after,
                    parent=prior['revision'] if prior else None,
                    expected=expected,
                    dependencies=[dest],
                    delta=delta,
                    history=history,
                )
            metrics.update(
                completed=bool(valid),
                artifact=str(dest if valid else source),
                history=history if valid else history[:-1],
                edit_s=CLOCK() - start,
            )
        if kind == '3dcodeedit':
            if os.environ.get('BENCH_DEFER_REVIEW') and valid:
                metrics['review_pending'] = True
            else:
                review(out, metrics, instruction, valid, err)
        metrics['feedback_s'] = CLOCK() - start
    except Exception as exc:
        (out / 'error.txt').write_text(traceback.format_exc())
        metrics['error'] = repr(exc)
    finally:
        metrics['elapsed_s'] = CLOCK() - start
        write_json(out / 'outcome.json', metrics)
    return metrics


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('root', type=Path)
    a.add_argument('stage')
    a.add_argument('kind')
    a.add_argument('case')
    a.add_argument('arm')
    a.add_argument('turn', type=int)
    p = a.parse_args()
    r = p.root.resolve()
    sys.path.insert(0, read(r / 'manifest.json')['roots']['code'])
    m = run(
        r,
        p.kind,
        p.case,
        p.arm,
        p.turn,
        p.stage,
        float(os.environ.get('BENCH_MONO', CLOCK())),
    )
    print(json.dumps(m, ensure_ascii=False), flush=True)
