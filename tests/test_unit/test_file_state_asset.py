from pathlib import Path

import numpy as np
import pytest
import trimesh
from embodied_gen.harness.asset_edit import gltf, program
from embodied_gen.harness.scene_edit.state_store import RevisionStore, stamps


def test_edit_preserves_sibling_and_resumes(tmp_path):
    src = tmp_path / 'src.glb'
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), node_name='body')
    scene.add_geometry(
        trimesh.creation.icosphere(),
        node_name='cap',
        transform=trimesh.transformations.translation_matrix([1, 2, 3]),
    )
    src.write_bytes(scene.export(file_type='glb'))
    g, _ = gltf.header(src)
    initial, _ = gltf.observe(g)
    cap = next(k for k, n in initial['nodes'].items() if n['name'] == 'cap')
    store = RevisionStore(tmp_path / 'session')
    with store.writer():
        rev = store.begin()
        dst = rev / 'artifact' / 'scene.glb'
        actions = [{'op': 'paint', 'id': cap, 'color': [1, 0, 0]}]
        updated, count = gltf.edit(src, dst, actions, initial, True)
        assert count == 0 and updated == gltf.observe(gltf.header(dst)[0])[0]
        mark = stamps([src])
        store.commit(
            rev,
            dst,
            updated,
            parent=None,
            expected=mark,
            dependencies=[dst],
            delta={},
            history=[],
        )
    current = RevisionStore(store.root).load()
    assert current['state'] == updated
    with store.writer():
        r = store.begin()
        out = r / 'artifact' / 'scene.glb'
        rest, n = gltf.edit(
            dst, out, [{'op': 'remove', 'id': cap}], current['state'], True
        )
        assert n == 0 and len(rest['nodes']) == 1
        assert rest == gltf.observe(gltf.header(out)[0])[0]
        with pytest.raises(RuntimeError):
            store.commit(
                r,
                out,
                rest,
                parent='wrong',
                expected=current['dependencies'],
                dependencies=[out],
                delta={},
                history=[],
            )
        assert store.current() == rev.name
    dst.touch()
    assert store.load()['state'] is None


def test_program_index_rebuild():
    assert program.observe('scale=3\ndef f(): return scale')['functions'] == {
        'f': ['scale']
    }


def test_failed_edit_and_external_mutation_never_advance_pointer(tmp_path):
    store = RevisionStore(tmp_path / 'session')
    asset = tmp_path / 'asset'
    asset.write_text('base')
    with store.writer():
        d = store.begin()
        store.commit(
            d,
            asset,
            {'schema': 1},
            parent=None,
            expected=stamps([asset]),
            dependencies=[asset],
            delta={},
            history=[],
        )
        original = store.load()
        candidate = store.begin()
        with pytest.raises(BlockingIOError):
            with RevisionStore(store.root).writer():
                pass
        asset.write_text('external')
        assert store.load()['state'] is None
        with pytest.raises(RuntimeError):
            store.commit(
                candidate,
                asset,
                {},
                parent=d.name,
                expected=original['dependencies'],
                dependencies=[asset],
                delta={},
                history=[],
            )
        assert store.current() == d.name
        (d / 'state.json').unlink()
        assert store.load()['state'] is None


def test_external_dependency_rename_move_and_id_loss(tmp_path):
    import xml.etree.ElementTree as ET

    from embodied_gen.harness.scene_edit.native_session import (
        NativeTurn,
        dependencies,
    )
    from embodied_gen.harness.scene_edit.planning import compact_state

    robot = ET.Element('robot', name='small')
    ET.SubElement(robot, 'link', name='base')
    f = tmp_path / 'model.obj'
    mesh = trimesh.creation.box()
    mesh.export(f)
    f.write_text('mtllib color.mtl\n' + f.read_text())
    tex = tmp_path / 'panel.png'
    from PIL import Image

    Image.new('RGB', (2, 2), 'white').save(tex)
    (tmp_path / 'color.mtl').write_text('newmtl panel\nmap_Kd panel.png\n')
    for name, xyz in [('study_0_floor', '0 0 -1'), ('lamp', '0 0 0')]:
        link = ET.SubElement(robot, 'link', name=name, scene_edit_id=name)
        for kind in ['visual', 'collision']:
            ET.SubElement(
                ET.SubElement(ET.SubElement(link, kind), 'geometry'),
                'mesh',
                filename=f.name,
            )
        j = ET.SubElement(robot, 'joint', type='fixed', name='j' + name)
        ET.SubElement(j, 'parent', link='base')
        ET.SubElement(j, 'child', link=name)
        ET.SubElement(j, 'origin', xyz=xyz, rpy='0 0 0')
    source = tmp_path / 'source.urdf'
    tree = ET.ElementTree(robot)
    tree.write(source)
    store = RevisionStore(tmp_path / 'session')
    with store.writer():
        rev = store.begin()
        store.commit(
            rev,
            source,
            dict(schema='bootstrap'),
            parent=None,
            expected=stamps([source]),
            dependencies=dependencies(source),
            delta={},
            history=[],
        )
        first = NativeTurn(store, store.load())
        first.publish(source, [{'instruction': 'observe'}], {}, 'noop')
        hit = NativeTurn(store, store.load())
        assert hit.metrics['state_hit'] and hit.before[2] == first.before[2]
        assert hit.manager(source).get_instance_names() == ['lamp']
        tex.touch()
        assert store.load()['state'] is None
        lamp = robot.findall('link')[-1]
        lamp.set('name', 'new')
        lamp.attrib.pop('scene_edit_id')
        joint = robot.findall('joint')[-1]
        joint.find('child').set('link', 'new')
        joint.find('origin').set('xyz', '1 0 .5')
        tree.write(source)
        refreshed = NativeTurn(store, store.load())
        entry = refreshed.before[2][0]
        assert (
            entry['id'] == 'new'
            and entry['xyz'] == [1.0, 0.0, 0.5]
            and entry['stable_id'] is None
        )
        state = compact_state(
            dict(
                facts=[entry],
                unidentified_objects=True,
                objects=['new'],
                rooms=[],
                assets=[],
                history=[
                    {
                        'status': 'execute',
                        'inserted': [
                            dict(
                                asset='lamp',
                                instance_key='lamp',
                                stable_id='lamp',
                            )
                        ],
                    }
                ],
            )
        )
        assert state['history'][0]['inserted'][0]['resolution'] == 'unverified'


def test_region_scores_noop_and_oracle():
    import sys
    from unittest.mock import patch

    with patch.object(
        sys,
        'path',
        [
            str(
                Path(__file__).resolve().parents[2] / 'scripts/file_state_edit'
            ),
            *sys.path,
        ],
    ):
        from evaluate import geometry_scores
    base = np.zeros((8, 8, 8), bool)
    base[:1] = True
    base[2:4] = True
    target = base.copy()
    target[2:4] = False
    target[4:6] = True
    region = base ^ target
    noop = geometry_scores(
        base, target, base, base, base, region, region, 'replace'
    )
    oracle = geometry_scores(
        base, target, base, target, base, region, region, 'replace'
    )
    assert noop['IF'] == 0 and oracle['IF'] == 1
    assert noop['CC_initial'] == oracle['CC_prev'] == 1
    assert (
        geometry_scores(
            base, target, target, target, base, region, region, 'remove'
        )['IF']
        is None
    )


def test_attachment_texture_identity_and_world_bounds(tmp_path):
    initial = tmp_path / 'src.glb'
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(),
        node_name='part',
        transform=trimesh.transformations.translation_matrix([4, 0, 0]),
    )
    initial.write_bytes(scene.export(file_type='glb'))
    before, _ = gltf.observe(gltf.header(initial)[0])
    pid = next(iter(before['nodes']))
    mid = tmp_path / 'with.glb'
    final = tmp_path / 'paint.glb'
    state, n = gltf.edit(
        initial,
        mid,
        [
            dict(
                op='add',
                name='cap',
                shape='box',
                center=[4, 1, 0],
                size=[0.4, 1, 0.4],
                color=[0.1, 0.2, 0.3],
                attach_to=int(pid),
            )
        ],
        before,
        True,
        initialize_ids=True,
    )
    cid = next(k for k in state['nodes'] if k != pid)
    node = state['nodes'][cid]
    assert (
        n == 1
        and node['parent'] == int(pid)
        and node['stable_id'] != state['nodes'][pid]['stable_id']
    )
    assert np.allclose(node['bounds'], [[3.8, 0.5, -0.2], [4.2, 1.5, 0.2]])
    painted, n = gltf.edit(
        mid,
        final,
        [
            dict(
                op='paint',
                id=int(cid),
                color=[1, 1, 1],
                pixels=[[[0, 255, 0], [255, 0, 0]]],
            )
        ],
        state,
        True,
    )
    assert n == 0 and painted == gltf.observe(gltf.header(final)[0])[0]
    assert painted['nodes'][cid]['stable_id'] == node['stable_id']
    assert painted['nodes'][cid]['materials'][0]['texture'] is not None
    # An externally duplicated UUID has no confirmed identity on either node.
    g, _ = gltf.header(final)
    g['nodes'][int(cid)]['extras']['scene_edit_id'] = state['nodes'][pid][
        'stable_id'
    ]
    assert all(
        x['stable_id'] is None for x in gltf.observe(g)[0]['nodes'].values()
    )


def test_initial_version_is_usable_after_first_request_crashes(tmp_path):
    from embodied_gen.harness.scene_edit.native_session import session_edit

    source = tmp_path / 'empty.urdf'
    source.write_text('<robot name="empty"><link name="base"/></robot>')

    class Abort:
        def edit(self, *args, **kwargs):
            raise RuntimeError('first instruction interrupted')

    with pytest.raises(RuntimeError, match='interrupted'):
        session_edit(
            Abort(), tmp_path / 's', source, 'edit', tmp_path / 'attempt'
        )
    store = RevisionStore(tmp_path / 's')
    old = store.load()
    assert (
        old['state']['facts'] == []
        and old['state']['schema'] == 'static-urdf-v1'
    )
    assert old['history'] == [] and old['artifact'] != source

    class Check:
        def edit(self, *args, session_turn, **kwargs):
            return session_turn.metrics['state_hit']

    assert session_edit(
        Check(), store.root, None, 'inspect', tmp_path / 'next'
    )
