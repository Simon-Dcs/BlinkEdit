"""Secondary control only: identical recorded F0 actions for delegated pilot instructions."""

import tempfile, sys, time
from pathlib import Path
from prepare import R, OLD, read, save

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util
from rules import eh, removed_doc, additions_doc
from embodied_gen.harness.scene_edit.state_store import RevisionStore, stamps
import trimesh


def main():
    spec = importlib.util.spec_from_file_location(
        'pilot_backend', OLD / 'setup/pilot-source/asset_edit/gltf.py'
    )
    gltf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gltf)
    m = read(OLD / 'manifest.json')
    res = []
    for key in m['dev']['edithero']:
        for arm in ('F0', 'F0R', 'F1R'):
            with tempfile.TemporaryDirectory(dir=R) as work:
                w = Path(work)
                src = OLD / 'setup/data/states' / key / 'turn00.glb'
                store = RevisionStore(w / 'session') if arm == 'F1R' else None
                import contextlib

                archive = {}
                added = []
                history = []
                for t, q in enumerate(m['chain_info'][key]['instructions'], 1):
                    original = (
                        OLD / 'edithero/dev' / key / 'F0' / f'turn{t:02}'
                    )
                    rr = read(original / 'outcome.json')
                    with store.writer() if store else contextlib.nullcontext():
                        ti = time.perf_counter()
                        prior = store.load() if store else None
                        if prior:
                            src = prior['artifact']
                        g = gltf.header(src)[0]
                        state = prior['state'] if prior else gltf.observe(g)[0]
                        local = (
                            eh(q, state, src, archive, added)
                            if arm != 'F0'
                            else None
                        )
                        actions = local or read(original / 'actions.json')
                        rev = store.begin() if store else None
                        dst = (
                            rev / 'artifact/out.glb' if rev else w / f'{t}.glb'
                        )
                        state2, n = gltf.edit(
                            src, dst, actions, state, store is not None
                        )
                        if not trimesh.load(
                            dst, force='scene', skip_materials=True
                        ).graph.nodes_geometry:
                            raise ValueError('empty')
                        if arm != 'F0':
                            archive = removed_doc(g, actions, state, archive)
                            added = additions_doc(
                                q, actions, state, state2, added
                            )
                        history = history + [q]
                        if store:
                            store.commit(
                                rev,
                                dst,
                                state2,
                                parent=prior['revision'] if prior else None,
                                expected=stamps([src]),
                                dependencies=[dst],
                                delta={'actions': actions},
                                history=history,
                            )
                        elapsed = time.perf_counter() - ti
                    # Identical parsed JSON and binary mesh/material payload permit exact reuse of prior offline scores.
                    ref = Path(rr['artifact'])
                    gd, od = gltf.header(dst)
                    gr, orr = gltf.header(ref)
                    with dst.open('rb') as a, ref.open('rb') as b:
                        a.seek(od)
                        b.seek(orr)
                        same = gd == gr and a.read() == b.read()
                    score = read(original / 'score.json') if same else {}
                    u = read(original / 'editor/request_00/record.json')[
                        'usage'
                    ]
                    res.append(
                        dict(
                            chain=key,
                            turn=t,
                            arm=arm,
                            route='local' if local else 'recorded_editor',
                            identical_artifact=same,
                            IF=score.get('IF'),
                            CC=score.get('CC_prev'),
                            local_s=elapsed,
                            recorded_tokens=0
                            if local
                            else u['input_tokens'] + u['output_tokens'],
                        )
                    )
                    src = dst
    save(R / 'metrics/development-replay.json', res)


if __name__ == '__main__':
    main()
