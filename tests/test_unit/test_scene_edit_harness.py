"""Native state regression, isolated policy execution and atomic scene edits."""

import importlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import box
from embodied_gen.harness.scene_edit.package import IsolatedPolicy, Versions
from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.runtime import (
    SceneEditor,
    native_manager,
    same_intent,
)
from embodied_gen.harness.scene_edit.scene import clone_scene, inventory

ROOT = Path(__file__).resolve().parents[2]
H0 = ROOT / 'harnesses/scene_edit_h0'
api = importlib.import_module(
    'embodied_gen.skills.spatial-computing.api.floorplan_api'
)
collector = importlib.import_module(
    'embodied_gen.skills.spatial-computing.core.collector'
)


def test_new_instance_metadata_matches_returned_height(tmp_path, monkeypatch):
    mesh = tmp_path / 'cube.obj'
    trimesh.creation.box([0.1, 0.1, 0.1]).export(mesh)
    c = collector.UrdfSemanticInfoCollector()
    monkeypatch.setattr(c, '_resolve_room_polygon', lambda _: None)
    monkeypatch.setattr(
        c,
        '_resolve_placement_target',
        lambda *a: (box(-2, -2, 2, 2), box(3, 3, 4, 4), 0.8),
    )
    monkeypatch.setattr(
        c,
        '_try_place_polygon',
        lambda *a: (0.2, 0.3, box(0.15, 0.25, 0.25, 0.35)),
    )
    xyz = c.add_instance(str(mesh), 'cube_a')
    assert xyz[2] > 0.8
    assert c.get_instance_center('cube_a')[2] == xyz[2]
    assert np.array_equal(c.instance_meta['cube_a']['xyz'], xyz)
    assert c.instance_meta['cube_a']['original_link_name'] == 'cube_a'


def test_failed_placement_does_not_register_metadata(tmp_path, monkeypatch):
    mesh = tmp_path / 'cube.obj'
    trimesh.creation.box([0.1, 0.1, 0.1]).export(mesh)
    c = collector.UrdfSemanticInfoCollector()
    monkeypatch.setattr(c, '_resolve_room_polygon', lambda _: None)
    monkeypatch.setattr(
        c,
        '_resolve_placement_target',
        lambda *a: (box(-2, -2, 2, 2), box(3, 3, 4, 4), 0.8),
    )
    monkeypatch.setattr(c, '_try_place_polygon', lambda *a: None)
    assert c.add_instance(str(mesh), 'cube_a') is None
    assert 'cube_a' not in c.instances and 'cube_a' not in c.instance_meta


def test_policy_mounts_hide_host_and_auth(tmp_path):
    policy = tmp_path / 'policy'
    shutil.copytree(H0, policy)
    (policy / 'policy.py').write_text('''import os, socket
class Policy:
 def prepare_request(self, payload, state):
  blocked=False
  try: open('/policy/test','w').write('x')
  except OSError: blocked=True
  open('/scratch/test','w').write('x')
  net=False
  try:
   s=socket.socket(); s.settimeout(.1); s.connect(('1.1.1.1',80)); net=True
  except OSError: pass
  return {'host':os.path.exists('/data2/dcs/EmbodiedGen'), 'home':os.path.exists('/home/dcs/.codex'), 'write_blocked':blocked,'network':net,'env':list(os.environ)}
''')
    with IsolatedPolicy(policy) as p:
        result = p.call('prepare_request', {})
    assert result['host'] is False and result['home'] is False
    assert result['write_blocked'] and not result['network']
    assert not any('KEY' in key or 'TOKEN' in key for key in result['env'])


def test_write_once_refuses_changed_request(tmp_path):
    p = tmp_path / 'request.json'
    write_json(p, {'commit': 'a'}, once=True)
    write_json(p, {'commit': 'a'}, once=True)
    with pytest.raises(ValueError):
        write_json(p, {'commit': 'b'}, once=True)


def test_version_parent_and_readonly_materialization(tmp_path):
    source = tmp_path / 'source'
    shutil.copytree(H0, source)
    store = Versions(tmp_path / 'versions')
    initial, _ = store.freeze(source, 'initial')
    (source / 'memory/MEMORY.md').write_text('Observed mechanism\n')
    child, frozen = store.freeze(source, 'child', initial)
    assert store.git('show', '-s', '--format=%P', child) == initial
    assert (frozen / 'memory/MEMORY.md').read_text() == 'Observed mechanism\n'
    assert (frozen / 'policy.py').stat().st_mode & 0o222 == 0


def test_execution_order_cannot_change_target():
    p = {
        'status': 'execute',
        'delete': ['lamp_a'],
        'insert': [
            {'asset': 'vase', 'instance_key': 'v', 'on_instance': 'table_a'}
        ],
    }
    assert same_intent(p, p)
    q = json.loads(json.dumps(p))
    q['delete'] = ['lamp_b']
    assert not same_intent(p, q)


def test_transport_notification_is_not_an_executable_model_tool():
    from embodied_gen.harness.scene_edit.model import tool_items

    events = [
        {'type': 'error', 'message': 'Reconnecting'},
        {
            'type': 'item.completed',
            'item': {'type': 'error', 'message': 'HTTPS fallback'},
        },
        {
            'type': 'item.completed',
            'item': {'type': 'agent_message', 'text': '{}'},
        },
    ]
    assert tool_items(events) == []
    events.append(
        {'type': 'item.completed', 'item': {'type': 'command_execution'}}
    )
    assert tool_items(events) == ['command_execution']


def test_declared_skill_is_called_and_retains_raw_capability_trace(tmp_path):
    policy = tmp_path / 'policy'
    shutil.copytree(H0, policy)
    manifest = json.loads((policy / 'HARNESS.json').read_text())
    manifest['skills'] = [
        {
            'name': 'inspect_geometry',
            'requires': {'inspect': 'scene.inspect@1'},
        }
    ]
    (policy / 'HARNESS.json').write_text(json.dumps(manifest))
    skill = policy / 'skills/inspect_geometry'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('Read public geometry before planning.\n')
    (skill / 'tool.py').write_text('''def build(runtime):
 return {'name':'inspect_geometry','description':'Read geometry','input_schema':{'type':'object'},'handler':lambda args:runtime.call('inspect',args)}
''')
    (policy / 'policy.py').write_text('''class Policy:
 def prepare_request(self,payload,state):
  return {'context':{'facts':state['skills']['inspect_geometry']({})},'guidance':'','placement_attempts':1}
''')
    with IsolatedPolicy(policy) as p:
        result = p.call(
            'prepare_request', {'facts': [{'id': 'box_a', 'xyz': [0, 0, 0.5]}]}
        )
        assert result['context']['facts'][0]['xyz'][2] == 0.5
        assert p.calls == [
            {
                'skill': 'inspect_geometry',
                'capability': 'scene.inspect@1',
                'args': {},
            }
        ]


def test_undeclared_capability_is_rejected(tmp_path):
    policy = tmp_path / 'policy'
    shutil.copytree(H0, policy)
    m = json.loads((policy / 'HARNESS.json').read_text())
    m['capabilities'].append('host.shell@1')
    (policy / 'HARNESS.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):
        IsolatedPolicy(policy)


@pytest.fixture
def tiny_scene(tmp_path):
    d = tmp_path / 'input'
    d.mkdir()
    robot = ET.Element('robot', name='tiny')
    ET.SubElement(robot, 'link', name='base')
    for name, size, xyz in [
        ('office_0_floor', [3, 3, 0.1], [0, 0, -0.05]),
        ('table_office_001', [1, 1, 0.8], [0, 0, 0.4]),
        ('lamp_office_001', [0.1, 0.1, 0.2], [1, 0, 0.1]),
    ]:
        mesh = d / (name + '.obj')
        trimesh.creation.box(size).export(mesh)
        link = ET.SubElement(robot, 'link', name=name)
        for kind in ['visual', 'collision']:
            part = ET.SubElement(link, kind)
            geo = ET.SubElement(part, 'geometry')
            ET.SubElement(geo, 'mesh', filename=mesh.name)
        joint = ET.SubElement(
            robot, 'joint', name='joint_' + name, type='fixed'
        )
        ET.SubElement(joint, 'parent', link='base')
        ET.SubElement(joint, 'child', link=name)
        ET.SubElement(
            joint, 'origin', xyz=' '.join(map(str, xyz)), rpy='0 0 0'
        )
    path = d / 'scene_updated.urdf'
    ET.ElementTree(robot).write(path)
    for name in ['vase', 'mug']:
        trimesh.creation.box([0.1, 0.1, 0.1]).export(d / (name + '.obj'))
        shutil.copy2(d / (name + '.obj'), d / (name + '_collision.obj'))
    return path, {'vase': str(d / 'vase.obj'), 'mug': str(d / 'mug.obj')}


def test_snapshot_has_dependency_closure(tiny_scene, tmp_path):
    path, _ = tiny_scene
    clone = clone_scene(path, tmp_path / 'clone')
    before, _, _ = inventory(path)
    after, _, _ = inventory(clone)
    assert set(before) == set(after)
    assert all(
        np.array_equal(before[k]['mesh'].vertices, after[k]['mesh'].vertices)
        for k in before
    )


def test_stable_identity_survives_an_external_rename(tiny_scene, tmp_path):
    path, _ = tiny_scene
    t = ET.parse(path)
    for link in t.findall('link'):
        if link.get('name') == 'lamp_office_001':
            link.set('name', 'renamed_object')
            link.set('scene_edit_id', 'identity_001')
    for j in t.findall('joint'):
        if j.find('child').get('link') == 'lamp_office_001':
            j.find('child').set('link', 'renamed_object')
    t.write(path)
    clone = clone_scene(path, tmp_path / 'clone')
    _, _, facts = inventory(clone)
    assert (
        next(f for f in facts if f['id'] == 'renamed_object')['stable_id']
        == 'identity_001'
    )


def test_opt_in_editor_keeps_physical_light_named_meshes(tiny_scene):
    path, _ = tiny_scene
    tree = ET.parse(path)
    for link in tree.findall('link'):
        if link.get('name') == 'lamp_office_001':
            link.set('name', 'light_office_001')
    for joint in tree.findall('joint'):
        if joint.find('child').get('link') == 'lamp_office_001':
            joint.find('child').set('link', 'light_office_001')
    tree.write(path)
    assert (
        'light_office_001'
        not in api.FloorplanManager(str(path)).get_instance_names()
    )
    manager = native_manager(path)
    assert 'light_office_001' in manager.get_instance_names()
    assert manager.delete_object('light_office_001')


class DummyRenderer:
    def reset(self):
        pass

    def preview(self, *args):
        return {}

    def gpu(self):
        return {'visible': '6'}

    def release(self, *args, **kw):
        return {}, []


def test_partial_batch_does_not_publish_or_overwrite_input(
    tiny_scene, tmp_path, monkeypatch
):
    path, catalog = tiny_scene
    before = path.read_bytes()
    plan = {
        'status': 'execute',
        'delete': ['lamp_office_001'],
        'insert': [
            {
                'asset': a,
                'instance_key': 'new_' + a,
                'on_instance': 'table_office_001',
                'in_room': 'office_floor',
                'place_strategy': 'top',
            }
            for a in ['vase', 'mug']
        ],
        'message': 'edit',
    }

    class FakeGateway:
        def __init__(self, *a, **kw):
            self.records = []

        def query(self, *a, **kw):
            return json.dumps(plan)

    import embodied_gen.harness.scene_edit.runtime as runtime

    monkeypatch.setattr(runtime, 'ModelGateway', FakeGateway)
    original = collector.UrdfSemanticInfoCollector.add_instance

    def fail_second(self, *args, **kw):
        if kw['instance_key'] == 'new_mug':
            return None
        return original(self, *args, **kw)

    monkeypatch.setattr(
        collector.UrdfSemanticInfoCollector, 'add_instance', fail_second
    )
    rec = SceneEditor(catalog, H0, renderer=DummyRenderer()).edit(
        'edit', path, tmp_path / 'edit'
    )
    assert (
        rec['status'] == 'failed'
        and rec['reason'] == 'native_search_exhausted'
    )
    assert rec['output'] == str(path) and not rec['published']
    assert not (tmp_path / 'edit/published.json').exists()
    assert path.read_bytes() == before
    partial = tmp_path / 'edit/candidate_0/scene_updated.urdf'
    assert 'new_vase' in [
        e.get('name') for e in ET.parse(partial).findall('link')
    ]
    assert rec['error'] is None


@pytest.mark.parametrize(
    'stable_ids,expected',
    [
        (['identity_1'], 'resolved'),
        ([], 'absent_identity'),
        (['identity_1', 'identity_1'], 'ambiguous_identity'),
    ],
)
def test_adopted_policy_resolves_identity_without_name_guessing(
    stable_ids, expected
):
    policy = ROOT / 'harnesses/scene_edit_vista'
    assert policy.is_dir(), (
        'Install the adopted frozen policy before this test'
    )
    facts = [
        {'id': f'renamed_{i}', 'stable_id': stable}
        for i, stable in enumerate(stable_ids)
    ]
    # Reusing a familiar name must not make this replacement the old object.
    facts.append({'id': 'old_mug', 'stable_id': 'unrelated_identity'})
    payload = dict(
        objects=[f['id'] for f in facts],
        rooms=[],
        assets=['mug'],
        facts=facts,
        history=[
            dict(
                status='execute',
                inserted=[
                    dict(
                        instance_key='old_mug',
                        stable_id='identity_1',
                        asset='mug',
                    )
                ],
            )
        ],
    )
    with IsolatedPolicy(policy) as worker:
        prepared = worker.call('prepare_request', payload)
    binding = prepared['context']['identity_bindings'][0]
    assert binding['resolution'] == expected
    assert (
        binding['current']['id'] == 'renamed_0'
        if expected == 'resolved'
        else binding['current'] is None
    )


def test_adopted_policy_refreshes_empty_scene_in_same_session():
    payload = dict(
        objects=['lamp_a'],
        rooms=[],
        assets=[],
        history=[],
        facts=[{'id': 'lamp_a', 'stable_id': None}],
    )
    with IsolatedPolicy(ROOT / 'harnesses/scene_edit_vista') as worker:
        assert worker.call('prepare_request', payload)['context']['facts']
        payload.update(objects=[], facts=[])
        refreshed = worker.call('prepare_request', payload)['context']
        assert refreshed['facts'] == [] and refreshed['objects'] == []
        assert refreshed['observation']['available'] is True
