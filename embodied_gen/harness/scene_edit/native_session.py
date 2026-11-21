"""Recover geometry and the native collector; only verified candidates become current."""

import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from shapely import from_wkb

from .rendering import parse
from .runtime import native_manager
from .scene import (
    asset_references,
    clone_scene,
    describe,
    inventory,
    native_name,
)
from .state_store import RevisionStore, stamps

SCHEMA = 'static-urdf-v1'


def validate_format(path):
    root = ET.parse(path)
    if any(
        j.get('type') != 'fixed' or j.find('parent').get('link') != 'base'
        for j in root.findall('joint')
    ):
        raise ValueError('Session mode requires fixed joints to base')
    if any(
        not np.array_equal(
            np.fromstring(m.get('scale', '1 1 1'), sep=' '), [1, 1, 1]
        )
        for m in root.findall('.//mesh')
    ):
        raise ValueError('Unsupported mesh scale')
    for link in root.findall('link'):
        if link.get('name') == 'base':
            continue
        for kind in ('visual', 'collision'):
            nodes = link.findall(kind)
            if len(nodes) != 1 or nodes[0].find('geometry/mesh') is None:
                raise ValueError(
                    'Session requires one visual and one collision mesh per link'
                )
            origin = nodes[0].find('origin')
            if origin is not None and any(
                np.any(np.fromstring(origin.get(k, '0 0 0'), sep=' '))
                for k in ('xyz', 'rpy')
            ):
                raise ValueError('Unsupported mesh-local transform')


def collectors(c):
    return dict(
        polygons={k: v.wkb_hex for k, v in c.instances.items()},
        links={
            k: m.get('original_link_name', k)
            for k, m in c.instance_meta.items()
        },
    )


def restore(c, path, snap):
    c.urdf_path = str(path)
    c._tree = ET.parse(path)
    c._root = c._tree.getroot()
    joints = c._build_link_transforms()
    links = {link.get('name'): link for link in c._tree.findall('link')}
    c.instances = {
        k: from_wkb(bytes.fromhex(v)) for k, v in snap['polygons'].items()
    }
    for k, name in snap['links'].items():
        link = links[name]
        xyz, rpy = joints[name]
        file = link.find('visual/geometry/mesh').get('filename')
        c.instance_meta[k] = dict(
            original_link_name=name,
            mesh_path=str(Path(path).parent / file),
            xyz=xyz,
            rpy=rpy,
        )
    c._update_internal_state()


class NativeTurn:
    def __init__(self, store, current, *, revision=None):
        self.store = store
        self.revision = revision or store.begin()
        self.current = current
        self.source = current['artifact']
        state = current['state']
        validate_format(self.source)
        self.blobs = {}
        self.copy_sources = {}
        self.clone_stamps = {}
        self.after = None
        hit = state is not None and state['schema'] == SCHEMA
        self.metrics = dict(
            state_hit=hit,
            meshes_loaded=0,
            meshes_reused=0,
            relations_recomputed=0,
        )
        if hit:

            def recover(name, path, xyz, rpy):
                blob = state['meshes'][name]
                self.blobs[name] = blob
                with np.load(blob, allow_pickle=False) as z:
                    return trimesh.Trimesh(z['v'], z['f'], process=False)

            obj = parse(self.source, geometry_only=True, cached_mesh=recover)
            self.before = obj, {native_name(k): k for k in obj}, state['facts']
            self.snapshot = state['collector']
            self.metrics['meshes_reused'] = len(obj)
        else:
            self.before = inventory(self.source, geometry_only=True)
            self.snapshot = collectors(native_manager(self.source).collector)
            self.metrics['meshes_loaded'] = len(self.before[0])
        self.after_snapshot = None

    def manager(self, path):
        snap = (
            self.after_snapshot
            if self.after and path == self.after[0]
            else self.snapshot
        )
        return native_manager(
            path, restore_collector=lambda c, p: restore(c, p, snap)
        )

    def clone(self, source, destination):
        def track(dst, src):
            self.copy_sources[dst.resolve()] = src

        path = clone_scene(source, destination, copied_from=track)
        self.clone_stamps = stamps(self.copy_sources)
        return path

    def inspect_candidate(self, path, plan, manager):
        before, mapping, facts = self.before
        reused = set()

        def reuse(key, file, xyz, rpy):
            p = file.resolve()
            s = before.get(key)
            if (
                s
                and self.copy_sources.get(p) == Path(s['visual'])
                and stamps([p])[str(p)] == self.clone_stamps[str(p)]
                and np.array_equal(xyz, s['xyz'])
                and np.array_equal(rpy, s['rpy'])
            ):
                reused.add(key)
                return s['mesh']

        after = parse(path, geometry_only=True, cached_mesh=reuse)
        changed = set(after) - reused
        removed = set(before) - set(after)
        retained = {}
        for f in facts:
            key = mapping[f['id']]
            if key not in reused:
                continue
            # A new support may meet an old object's bottom. Recompute it in that case.
            bottom = after[key]['mesh'].bounds[0, 2]
            if any(
                -0.003 <= bottom - after[n]['mesh'].bounds[1, 2] <= 0.03
                for n in changed
            ):
                continue
            retained[key] = dict(
                f,
                support=[s for s in f['support'] if mapping[s] not in removed],
            )
        result = describe(after, retained)
        self.metrics['meshes_reused'] += len(reused)
        self.metrics['meshes_loaded'] += len(changed)
        self.metrics['relations_recomputed'] += len(result[2]) - len(retained)
        self.after = (path, result, reused)
        self.after_snapshot = collectors(manager.collector)
        return result

    def publish(self, path, history, plan, status, *, checked=None):
        updated = self.after and path == self.after[0]
        obj, _, facts = self.after[1] if updated else self.before
        keep = self.after[2] if updated else set(obj)
        blobs = {}
        for i, (key, item) in enumerate(obj.items()):
            if key in keep and key in self.blobs:
                blobs[key] = self.blobs[key]
                continue
            dst = self.revision / f'geometry-{i}.npz'
            np.savez(
                dst,
                v=np.asarray(item['mesh'].vertices),
                f=np.asarray(item['mesh'].faces),
            )
            blobs[key] = str(dst)
        state = dict(
            schema=SCHEMA,
            units='metres',
            coordinates='world Z up; xyz/rpy radians',
            facts=facts,
            meshes=blobs,
            collector=self.after_snapshot if updated else self.snapshot,
        )
        # All assets of a private URDF are copied into its directory. Monitor its own
        # files (including newly referenced dependencies after explicit external edits).
        if updated:
            # clone_scene recorded each copied material dependency. New native assets
            # live in dedicated instance directories, not arbitrary external paths.
            deps = {path, *self.copy_sources}
            for item in plan['insert']:
                deps.update(
                    p
                    for p in (path.parent / item['instance_key']).rglob('*')
                    if p.is_file()
                )
        else:
            deps = set(self.current['dependencies'])
        self.store.commit(
            self.revision,
            path,
            state,
            parent=self.current['revision'],
            expected=checked
            if checked is not None
            else self.current['dependencies'],
            dependencies=deps,
            history=history,
            derived=blobs.values(),
            delta=dict(
                plan=plan,
                status=status,
                instruction=history[-1]['instruction'] if history else None,
            ),
        )
        self.blobs = blobs


def dependencies(path):
    """Read dependency declarations on a rebuild; hot loads stat the recorded paths."""

    files = {Path(path).resolve()}

    def visit(p):
        p = p.resolve()
        if p in files:
            return
        files.add(p)
        for ref in asset_references(p):
            visit(p.parent / ref)

    tree = ET.parse(path)
    for m in [
        *tree.findall('.//mesh'),
        *tree.findall('.//texture[@filename]'),
    ]:
        visit(path.parent / m.get('filename'))
    return files


def session_edit(editor, session, source, instruction, output, **kwargs):
    store = RevisionStore(session)
    with store.writer():
        cur = store.load()
        if cur and source:
            raise ValueError(
                'Resume with --session alone; use a fresh directory to switch scenes'
            )
        if not cur:
            if not source:
                raise ValueError('First turn needs --scene')
            source = Path(source).resolve()
            validate_format(source)
            deps = dependencies(source)
            expected = stamps(deps)
            rev = store.begin()
            path = clone_scene(source, rev / 'artifact' / 'initial')
            tree = ET.parse(path)
            for link in tree.findall('link'):
                if link.get('name') != 'base' and not link.get(
                    'scene_edit_id'
                ):
                    link.set('scene_edit_id', uuid.uuid4().hex)
            tree.write(path, encoding='utf-8', xml_declaration=True)
            cur = dict(
                revision=None,
                artifact=path,
                history=[],
                state=None,
                dependencies=stamps(dependencies(path)),
            )
            turn = NativeTurn(store, cur, revision=rev)
            turn.publish(
                path, [], {'import': str(source)}, 'import', checked=expected
            )
            cur = store.load()
            turn.current = cur
            turn.revision = store.begin()
        else:
            if cur['state'] is None:
                cur['dependencies'] = stamps(dependencies(cur['artifact']))
            turn = NativeTurn(store, cur)
        return editor.edit(
            instruction,
            cur['artifact'],
            output,
            history=cur['history'],
            session_turn=turn,
            **kwargs,
        )
