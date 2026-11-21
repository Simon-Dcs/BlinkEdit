"""Evidence -> live inspection -> model-written policy -> frozen autonomous test."""

import argparse
import ast
import json
import shutil
import time
from pathlib import Path

from run import check_budget, consumption, evaluate, evaluate_pair, select
from embodied_gen.harness.scene_edit.model import ModelGateway
from embodied_gen.harness.scene_edit.package import (
    IsolatedPolicy,
    Versions,
    package_files,
)
from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.rendering import Renderer
from embodied_gen.harness.scene_edit.scene import clone_scene, inventory

CONTRACT = '''Develop an executable EmbodiedGen editing policy; do not change model weights, host, evaluator or budgets.
Every new natural-language instruction is parsed by the existing model. Policy prepares factual context and may reorder already parsed insertions, but may not substitute targets, assets, scale or rooms.
The isolated package has stdlib Python only, no host files, credentials, network, geometry engine or evaluator access.
Policy hooks are synchronous:
prepare_request(payload,state) -> {"context": object, "guidance": string, "placement_attempts": 1 or 2}.
payload has instruction, objects, rooms, assets, confirmed history, and facts. Facts contain id, asset, room, xyz (metres), world bounds (metres), support IDs, collision, stable_id.
Newly inserted objects get stable_id persisted in the URDF and in history.inserted. An external editor can rename/move a link while retaining this identity. Current facts take precedence over old names. Deleted objects must not be rebound to other objects.
decide({plan,facts,history},state) -> plan. Only insertion order may change.
process_tool_result({plan,error,attempt},state) -> {retry:boolean,plan:plan,message:string}.
One attempt has 10000 surface samples and 2000 search iterations per object. Choosing 2 divides both totals equally across the two attempts, rather than increasing total budget. Each retry starts from an independent original-state copy.
HARNESS.json exact keys: schema_version=1, id, capabilities, policy="policy.py:Policy", skills (list), memory (relative file).
Host capabilities are scene.inspect@1, scene.query@1, scene.batch_edit@1, scene.preview@1. Preparation skills may declare only inspect/query in requires, e.g. {"name":"grounded_state","requires":{"inspect":"scene.inspect@1"}}.
Each skill has skills/<name>/SKILL.md and tool.py. tool.py exports build(runtime) returning exactly {name,description,input_schema,handler}; handler(args) can call runtime.call(alias,args). inspect returns current public facts, query takes {id} and returns one fact. The policy invokes the bound handler via state['skills'][name]({}). Calls are recorded.
state['memory_text'] contains the package's readonly experience. Keep scoped, operational rules and failure mechanisms, not episode IDs, literal object IDs, hidden expected answers, task-string lookup branches or coordinates.
Implement a reusable observation/identity/recovery mechanism, not only prompt wording. Use a declared skill when adding a mechanism. Inspect actual traces and live evidence. Preserve successful prior behavior. Do not copy test answers into package files.
'''


def package_text(path):
    return {
        name: (Path(path) / name).read_text() for name in package_files(path)
    }


def history_evidence(root, summary, limit=4):
    candidates = []
    failed = [e for e in summary['episodes'] if not e['passed']]
    chosen = failed[:3] + [e for e in summary['episodes'] if e['passed']][:1]
    for episode in chosen[:limit]:
        d = Path(root) / 'episodes' / summary['id'] / episode['id']
        for td in sorted(d.glob('turn_*')):
            result = next(td.glob('attempt_*/result.json'))
            r = json.loads(result.read_text())
            public = json.loads(
                (result.parent / 'public_state.json').read_text()
            )
            candidates.append(
                dict(
                    case=episode['id'],
                    turn=r['identity']['turn'],
                    instruction=r['instruction'],
                    passed=r.get('evaluation', {}).get('passed', False),
                    public_state=public,
                    actual_plan=r.get('plan'),
                    actual_status=r['status'],
                    error=r['error'],
                    guard=r.get('guard'),
                    verified_s=r['verified_s'],
                    llm_rpc_s=r['llm_rpc_s'],
                )
            )
    return candidates


def live_inspect(root, summary, case_ids, directory):
    from embodied_gen.harness.scene_edit.runtime import native_manager

    cases = {
        c['id']: c
        for c in json.loads((Path(root) / 'inputs/cases.json').read_text())
        if c['split'] == 'selection'
    }
    observations = []
    images = []
    renderer = Renderer(4)
    for case_id in case_ids[:2]:
        if case_id not in cases:
            raise ValueError('Live case must be a public selection case')
        # Rebuild a live copy at the relevant actual decision boundary.
        dirs = sorted(
            (Path(root) / 'episodes' / summary['id'] / case_id).glob('turn_*')
        )
        boundary = dirs[-1] if cases[case_id]['task'] == 'D2' else dirs[0]
        r = json.loads(
            next(boundary.glob('attempt_*/result.json')).read_text()
        )
        d = Path(directory) / case_id
        d.mkdir(parents=True)
        t = time.perf_counter()
        source = Path(r['input'])
        mode = 'actual_decision_boundary'
        failed_candidates = list(
            next(boundary.glob('attempt_*/result.json')).parent.glob(
                'candidate_*/scene_updated.urdf'
            )
        )
        if r['status'] == 'failed' and failed_candidates:
            source = failed_candidates[-1]
            mode = 'unpublished_failed_candidate'
        path = clone_scene(source, d / 'scene')
        objects, mapping, facts = inventory(path)
        manager = native_manager(path)
        positions = {
            key: manager.query_instance_center(key)
            for key in manager.get_instance_names()
        }
        renderer.reset()
        renderer.preview(path, d, 'live')
        item = dict(
            case=case_id,
            instruction=r['instruction'],
            facts=facts,
            native_queries=positions,
            controller='diagnostic',
            mode=mode,
            seconds=time.perf_counter() - t,
            scene=str(path),
            gpu=renderer.gpu(),
        )
        write_json(d / 'observation.json', item)
        observations.append(item)
        images.append(str(d / 'live_office.png'))
    # Explicitly release native GPU objects before the next measured task.
    renderer.reset()
    del renderer
    import gc

    gc.collect()
    return observations, images


def write_candidate(parent, workspace, updates):
    workspace = Path(workspace)
    shutil.copytree(parent, workspace)
    for p in workspace.rglob('*'):
        p.chmod(0o755 if p.is_dir() else 0o644)
    workspace.chmod(0o755)
    apply_updates(workspace, updates)
    package_files(workspace)


def apply_updates(workspace, updates):
    for relative, text in updates.items():
        path = Path(relative)
        if (
            path.is_absolute()
            or '..' in path.parts
            or '.git' in path.parts
            or (
                relative not in ['HARNESS.json', 'policy.py']
                and path.parts[0] not in ['skills', 'memory', 'prompts']
            )
        ):
            raise ValueError(
                'Candidate attempted to modify a host path: ' + relative
            )
        if not isinstance(text, str):
            raise ValueError('Candidate file content must be UTF-8 text')
        dest = Path(workspace) / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)


def remaining_supports_round(root):
    c = consumption(root)
    # Reserve the full 64-turn holdout, development and correction allowance.
    mean = (c['known_tokens'] / c['requests']) if c['requests'] else 20000
    required_requests = 32 + 64 + 12
    return (
        c['requests'] + required_requests <= 240
        and c['known_tokens'] + mean * required_requests <= 5000000
    )


def reject_memorized_ids(root, workspace):
    known = set()
    for p in (Path(root) / 'episodes/formal_selection_H0').glob(
        '*/turn_*/attempt_*/public_state.json'
    ):
        public = json.loads(p.read_text())
        known.update(public['objects'])
        known.update(public['rooms'])
    for name in package_files(workspace):
        if name.endswith('.py'):
            for node in ast.walk(
                ast.parse((Path(workspace) / name).read_text())
            ):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in known
                ):
                    raise ValueError(
                        'A generated policy memorized a scene identifier: '
                        + node.value
                    )
        elif name.startswith('memory/'):
            text = (Path(workspace) / name).read_text()
            if any(identifier in text for identifier in known):
                raise ValueError(
                    'Memory must store a general mechanism rather than a literal scene ID'
                )


def round_develop(root, round_index, parent, baseline):
    root = Path(root)
    d = root / 'development' / f'round_{round_index}'
    d.mkdir(parents=True, exist_ok=True)
    write_json(
        d / 'request.json',
        dict(
            parent_commit=parent['commit'],
            baseline=baseline['id'],
            round=round_index,
        ),
        once=True,
    )
    if (d / 'candidate.json').exists():
        candidate = json.loads((d / 'candidate.json').read_text())
        evaluate(
            root,
            candidate['id'],
            candidate['path'],
            candidate['commit'],
            purpose='development',
            cells=['D1_v0', 'T1_v0', 'D2_v0'],
        )
        formal = evaluate(
            root, candidate['id'], candidate['path'], candidate['commit']
        )
        return candidate, formal, select(root, baseline, formal)
    g = ModelGateway(
        d / 'meta_requests', identity=dict(phase='meta', round=round_index)
    )
    g.records = [
        json.loads(p.read_text())
        for p in sorted((d / 'meta_requests').glob('request_*/record.json'))
    ]
    evidence = history_evidence(root, baseline)
    previous = []
    for p in sorted((root / 'development').glob('round_*/candidate.json')):
        candidate = json.loads(p.read_text())
        selection = root / 'selections' / f"{candidate['id']}.json"
        if selection.exists():
            decision = json.loads(selection.read_text())
            trial = json.loads(
                (
                    root
                    / 'evaluations'
                    / f"formal_selection_{candidate['id']}"
                    / 'summary.json'
                ).read_text()
            )
            previous.append(
                dict(
                    candidate=candidate,
                    package=package_text(candidate['path']),
                    decision=decision,
                    evidence=history_evidence(root, trial, limit=2),
                )
            )
    intro = (
        CONTRACT
        + '\nCurrent package:\n'
        + json.dumps(package_text(parent['path']), ensure_ascii=False)
        + '\nActual execution evidence (no hidden answer fields):\n'
        + json.dumps(evidence, ensure_ascii=False)
        + '\nPrevious trials, including rejected changes and counterevidence:\n'
        + json.dumps(previous, ensure_ascii=False)
    )
    check_budget(root)
    if (d / 'hypothesis.json').exists():
        hypothesis = json.loads((d / 'hypothesis.json').read_text())
    else:
        hypothesis = json.loads(
            g.query(
                intro
                + '\nReturn JSON only: {"hypothesis":"one falsifiable mechanism","live_cases":["up to two case IDs visible in evidence"],"inspection":"what the live observations will distinguish"}. Choose two distinct scenes when possible.',
                stage='meta_hypothesis',
            )
        )
        write_json(d / 'hypothesis.json', hypothesis, once=True)
    # Live simulator work runs in a separate process so all GPU contexts close
    # before timed editing workers are admitted.
    live_job = d / 'live_job.json'
    write_json(
        live_job,
        dict(
            root=str(root),
            summary=baseline,
            case_ids=hypothesis['live_cases'],
            directory=str(d / 'live'),
        ),
    )
    import subprocess
    import sys

    if not (d / 'live_result.json').exists():
        check_budget(root)
        with (d / 'live_console.log').open('w') as log:
            subprocess.run(
                [sys.executable, __file__, 'live', '--job', str(live_job)],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=True,
            )
    live = json.loads((d / 'live_result.json').read_text())
    prompt = (
        intro
        + '\nHypothesis:\n'
        + json.dumps(hypothesis, ensure_ascii=False)
        + '\nLive observations:\n'
        + json.dumps(live['observations'], ensure_ascii=False)
        + '\nReturn JSON only: {"hypothesis":"mechanism","expected_effect":"prediction","files":{"relative path":"complete new file text"}}. Write an executable policy and at least one declared reusable skill. Preserve existing interfaces; no literal task/object IDs. Use current physical support/bounds/room facts and stable identity where needed. Include all modifications, not a prose suggestion.'
    )
    check_budget(root)
    if (d / 'proposal.json').exists():
        proposal = json.loads((d / 'proposal.json').read_text())
    else:
        proposal = json.loads(
            g.query(prompt, images=live['images'], stage='meta_implement')
        )
        write_json(d / 'proposal.json', proposal, once=True)
    workspace = d / 'workspace'
    if not workspace.exists():
        write_candidate(parent['path'], workspace, proposal['files'])
    # A load/bind check on public dummy facts catches syntax and ABI failures
    # before spending an autonomous scene evaluation.
    dummy = dict(
        instruction='在桌上放一个花瓶。',
        objects=['desk'],
        rooms=['room_floor'],
        assets=['vase'],
        history=[],
        facts=[
            dict(
                id='desk',
                asset='table',
                room='room_floor',
                xyz=[0, 0, 0.4],
                bounds=[[-0.5, -0.5, 0], [0.5, 0.5, 0.8]],
                support=['room_floor'],
                collision=True,
                stable_id=None,
            )
        ],
    )
    try:
        reject_memorized_ids(root, workspace)
        with IsolatedPolicy(workspace) as policy:
            preview = policy.call('prepare_request', dummy)
    except Exception as exc:
        write_json(d / 'bind_error.json', dict(error=repr(exc)))
        if len(g.records) >= 4:
            raise
        correction = json.loads(
            g.query(
                prompt
                + '\nPrevious proposal:\n'
                + json.dumps(proposal, ensure_ascii=False)
                + '\nBinding failed: '
                + repr(exc)
                + '. Return corrected complete files.',
                stage='meta_correct',
            )
        )
        write_json(d / 'correction.json', correction, once=True)
        # Only this round's generated workspace is replaced; parent is immutable.
        apply_updates(workspace, correction['files'])
        reject_memorized_ids(root, workspace)
        with IsolatedPolicy(workspace) as policy:
            preview = policy.call('prepare_request', dummy)
    write_json(d / 'binding_preview.json', preview)
    store = Versions(root / 'harness_repo')
    commit, frozen = store.freeze(
        workspace,
        f'Evidence-driven candidate H{round_index}',
        parent['commit'],
    )
    candidate = dict(
        id=f'H{round_index}',
        commit=commit,
        path=str(frozen),
        hypothesis=hypothesis['hypothesis'],
    )
    write_json(d / 'candidate.json', candidate, once=True)
    # Four timed development turns: ambiguous target, prior success, and a
    # two-turn external-update episode. Formal evaluation uses a new ID.
    evaluate(
        root,
        candidate['id'],
        frozen,
        commit,
        purpose='development',
        cells=['D1_v0', 'T1_v0', 'D2_v0'],
    )
    formal = evaluate(root, candidate['id'], frozen, commit)
    comparison = select(root, baseline, formal)
    return candidate, formal, comparison


def campaign(root):
    root = Path(root)
    h0 = json.loads((root / 'H0.json').read_text())
    h0['id'] = 'H0'
    initial = json.loads(
        (root / 'evaluations/formal_selection_H0/summary.json').read_text()
    )
    parent, selected = h0, initial
    decisions = []
    for round_index in range(1, 4):
        # A proposal review can end a redundant branch without turning a
        # partial matrix into a performance or promotion claim.
        if (
            root
            / 'development'
            / f'round_{round_index}'
            / 'proposal_review.json'
        ).exists():
            break
        existing = (
            root / 'development' / f'round_{round_index}' / 'candidate.json'
        )
        if not existing.exists() and not remaining_supports_round(root):
            break
        candidate, summary, decision = round_develop(
            root, round_index, parent, selected
        )
        decisions.append(decision)
        if decision['accepted']:
            parent, selected = candidate, summary
        write_json(
            root / 'selected.json',
            dict(**parent, selection_evaluation=selected['id']),
        )
    if parent['id'] == 'H0':
        # Still compare the latest valid candidate on U; a rejected version is
        # not silently presented as an adopted configuration.
        latest = (
            root / 'development' / f'round_{len(decisions)}' / 'candidate.json'
        )
        candidate = json.loads(latest.read_text()) if latest.exists() else None
    else:
        candidate = parent
    recommendation = h0
    if candidate is not None:
        left, right = evaluate_pair(root, h0, candidate)
        regressions = [
            e['id']
            for e in left['episodes']
            if e['passed']
            and not next(
                c['passed'] for c in right['episodes'] if c['id'] == e['id']
            )
        ]
        if (
            parent['id'] != 'H0'
            and not regressions
            and right['critical'] == 0
            and right['successes'] >= left['successes']
        ):
            recommendation = parent
        write_json(
            root / 'final_comparison.json',
            dict(
                H0=left,
                candidate=right,
                candidate_id=candidate['id'],
                selection_accepted=parent['id'] != 'H0',
                holdout_regressions=regressions,
                recommended=recommendation['id'],
            ),
        )
    write_json(root / 'recommended.json', recommendation, once=True)
    write_json(
        root / 'campaign_complete.json',
        dict(
            selected=parent,
            decisions=decisions,
            consumption={
                k: v for k, v in consumption(root).items() if k != 'records'
            },
        ),
        once=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['campaign', 'live'])
    p.add_argument('--root')
    p.add_argument('--job')
    a = p.parse_args()
    if a.action == 'live':
        job = json.loads(Path(a.job).read_text())
        observations, images = live_inspect(
            job['root'], job['summary'], job['case_ids'], job['directory']
        )
        write_json(
            Path(job['directory']).parent / 'live_result.json',
            dict(observations=observations, images=images),
        )
    else:
        campaign(
            Path(
                a.root
                or Path('outputs/vista_evolution/latest_run.txt')
                .read_text()
                .strip()
            )
        )


if __name__ == '__main__':
    main()
