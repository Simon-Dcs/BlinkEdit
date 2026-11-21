"""Matched local replay: same initial asset, same op list, no model sampling."""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from embodied_gen.harness.asset_edit import gltf
from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.state_store import (
    RevisionStore,
    read,
    stamps,
)


def replay(root, key, arm, directory):
    import trimesh

    group = root / 'edithero/dev' / key / 'F0'
    source = root / 'setup/data/states' / key / 'turn00.glb'
    ops = [read(p) for p in sorted(group.glob('turn*/actions.json'))]
    store = RevisionStore(directory / 'session') if arm == 'F1' else None
    start = time.perf_counter()
    counts = []
    for i, actions in enumerate(ops):
        begin = time.perf_counter()
        if store:
            with store.writer():
                prior = store.load()
                before = (
                    prior['state']
                    if prior
                    else gltf.observe(gltf.header(source)[0])[0]
                )
                expected = prior['dependencies'] if prior else stamps([source])
                rev = store.begin()
                dst = rev / 'artifact/object.glb'
                after, n = gltf.edit(source, dst, actions, before, True)
                if not trimesh.load(
                    dst, force='scene', skip_materials=True
                ).graph.nodes_geometry:
                    raise ValueError('empty')
                store.commit(
                    rev,
                    dst,
                    after,
                    parent=prior['revision'] if prior else None,
                    expected=expected,
                    dependencies=[dst],
                    delta={'ops': actions},
                    history=list(range(i + 1)),
                )
        else:
            before = gltf.observe(gltf.header(source)[0])[0]
            dst = directory / f'turn{i}.glb'
            after, n = gltf.edit(source, dst, actions, before, False)
            if not trimesh.load(
                dst, force='scene', skip_materials=True
            ).graph.nodes_geometry:
                raise ValueError('empty')
        counts.append(
            dict(turn=i + 1, recomputed=n, seconds=time.perf_counter() - begin)
        )
        source = dst
    return dict(
        seconds=time.perf_counter() - start,
        turns=counts,
        final_nodes=len(after['nodes']),
    )


def main(root):
    from embodied_gen.harness.scene_edit.native_session import NativeTurn
    from embodied_gen.harness.scene_edit.runtime import native_manager
    from embodied_gen.harness.scene_edit.scene import inventory

    cases = read(root / 'manifest.json')['dev']['edithero']
    out = []
    for repeat in range(5):
        for i, key in enumerate(cases):
            for arm in ['F1', 'F0'] if (repeat + i) % 2 else ['F0', 'F1']:
                with tempfile.TemporaryDirectory(dir=root) as w:
                    out.append(
                        dict(
                            repeat=repeat,
                            case=key,
                            arm=arm,
                            **replay(root, key, arm, Path(w)),
                        )
                    )
    store = RevisionStore(root / 'setup/native/session')
    path = store.load()['artifact']
    native = []
    for repeat in range(4):
        for arm in ['F1', 'F0'] if repeat % 2 else ['F0', 'F1']:
            with tempfile.TemporaryDirectory(dir=root) as w:
                scratch = RevisionStore(w)
                t = time.perf_counter()
                if arm == 'F1':
                    turn = NativeTurn(scratch, store.load())
                    _, _, facts = turn.before
                    manager = turn.manager(path)
                else:
                    _, _, facts = inventory(path, geometry_only=True)
                    manager = native_manager(path)
                elapsed = time.perf_counter() - t
                public = dict(
                    facts=facts,
                    rooms=sorted(manager.get_room_names()),
                    names=sorted(manager.get_instance_names()),
                )
                if not native:
                    reference = public
                assert public == reference
                native.append(dict(arm=arm, repeat=repeat, seconds=elapsed))
    write_json(
        root / 'metrics/local-replay.json', dict(edithero=out, native=native)
    )
    for group in [out, native]:
        for key in ['F0', 'F1']:
            print(
                key,
                statistics.mean(
                    row['seconds'] for row in group if row['arm'] == key
                ),
            )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    main(a.root.resolve())
