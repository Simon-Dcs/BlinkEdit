"""Scene snapshots and public geometry facts; no task answers or evaluator state."""

import shlex
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import MultiPoint

from .rendering import parse


def asset_references(path):
    """Read only OBJ/MTL declarations, not every vertex as a shell token list."""
    if path.suffix.lower() not in {'.obj', '.mtl'}:
        return
    with path.open(errors='replace') as stream:
        for line in stream:
            directive = line.lstrip().split(None, 1)[:1]
            if not directive:
                continue
            key = directive[0].lower()
            if key == 'mtllib' or (
                path.suffix.lower() == '.mtl'
                and key.startswith(('map_', 'bump', 'disp', 'decal'))
            ):
                words = shlex.split(line, comments=True)
                if key == 'mtllib':
                    yield from words[1:]
                elif len(words) > 1:
                    yield words[-1]


def clone_scene(source, destination, *, copied_from=None):
    """Copy the URDF and its mesh/material dependency closure, not its run tree."""
    source, destination = Path(source).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(source)
    copied = {}

    def resource(src, dst, root):
        src = src.resolve()
        if not src.is_file():
            raise ValueError(f"Missing asset dependency: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copied_from is not None:
            copied_from(dst, src)
        shutil.copy2(src, dst)
        for dep in asset_references(src):
            target = (dst.parent / dep).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(
                    'Asset material path escapes its mesh directory'
                )
            resource(src.parent / dep, target, root)

    for node in [
        *tree.findall('.//mesh'),
        *tree.findall('.//texture[@filename]'),
    ]:
        old = (source.parent / node.get('filename')).resolve()
        if old not in copied:
            folder = destination / 'assets' / f'mesh_{len(copied):03d}'
            resource(old, folder / old.name, folder)
            copied[old] = folder / old.name
        node.set('filename', str(copied[old].relative_to(destination)))
    result = destination / 'scene.urdf'
    tree.write(result, encoding='utf-8', xml_declaration=True)
    return result


def native_name(name):
    if name.endswith('_floor'):
        return '_'.join(name.split('_')[:-2] + ['floor'])
    return name.lower()


def inventory(path, *, geometry_only=False):
    objects = parse(Path(path), geometry_only=geometry_only)
    return describe(objects)


def describe(objects, retained=None):
    """Compute relations only for objects without a certified retained fact."""
    mapping = {native_name(k): k for k in objects}
    facts = []
    floors = {
        native_name(k): v for k, v in objects.items() if k.endswith('_floor')
    }
    for key, info in objects.items():
        if key.endswith('_floor'):
            continue
        if retained and key in retained:
            facts.append(retained[key])
            continue
        mesh = info['mesh']
        hull = MultiPoint(mesh.vertices[:, :2]).convex_hull
        room = next(
            (
                name
                for name, floor in floors.items()
                if MultiPoint(floor['mesh'].vertices[:, :2])
                .convex_hull.buffer(0.001)
                .covers(hull.representative_point())
            ),
            None,
        )
        supports = []
        for other, support in objects.items():
            if other == key:
                continue
            gap = mesh.bounds[0, 2] - support['mesh'].bounds[1, 2]
            if -0.003 <= gap <= 0.03 and MultiPoint(
                support['mesh'].vertices[:, :2]
            ).convex_hull.buffer(0.002).covers(hull):
                supports.append(native_name(other))
        facts.append(
            dict(
                id=native_name(key),
                asset=info['asset'],
                room=room,
                xyz=info['xyz'].tolist(),
                bounds=mesh.bounds.round(5).tolist(),
                support=supports,
                collision=info['collision'] is not None,
                stable_id=info['link'].get('scene_edit_id'),
            )
        )
    return objects, mapping, sorted(facts, key=lambda f: f['id'])


def guard(before, after, plan, mapping):
    """Public invariants for a declared edit, independent of task-specific truth."""
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    expected_removed = sorted(mapping[k] for k in plan['delete'])
    expected_added = sorted(i['instance_key'] for i in plan['insert'])
    same = all(
        np.array_equal(before[k]['xyz'], after[k]['xyz'])
        and np.array_equal(before[k]['rpy'], after[k]['rpy'])
        and np.array_equal(
            before[k]['mesh'].vertices, after[k]['mesh'].vertices
        )
        and Path(before[k]['collision']).read_bytes()
        == Path(after[k]['collision']).read_bytes()
        for k in set(before) & set(after)
    )
    details = []
    supports = {}
    for item in plan['insert']:
        name = item['instance_key']
        if name not in after:
            continue
        parent = mapping[item['on_instance']]
        supports[name] = parent
        mesh, support = after[name]['mesh'], after[parent]['mesh']
        hull = MultiPoint(mesh.vertices[:, :2]).convex_hull
        gap = float(mesh.bounds[0, 2] - support.bounds[1, 2])
        within = (
            MultiPoint(support.vertices[:, :2])
            .convex_hull.buffer(0.002)
            .covers(hull)
        )
        overlaps = []
        for other, info in after.items():
            if other in [name, parent] or other.endswith('_floor'):
                continue
            b = info['mesh'].bounds
            if (
                min(mesh.bounds[1, 2], b[1, 2])
                - max(mesh.bounds[0, 2], b[0, 2])
                <= 0.002
            ):
                continue
            area = hull.intersection(
                MultiPoint(info['mesh'].vertices[:, :2]).convex_hull
            ).area
            if area > 1e-5:
                overlaps.append(dict(id=other, area_m2=float(area)))
        details.append(
            dict(
                id=name,
                within=bool(within),
                gap_m=gap,
                overlaps=overlaps,
                collision=after[name]['collision'] is not None,
            )
        )
    passed = (
        removed == expected_removed
        and added == expected_added
        and same
        and all(
            d['within']
            and -0.002 <= d['gap_m'] <= 0.03
            and not d['overlaps']
            and d['collision']
            for d in details
        )
    )
    return dict(
        passed=bool(passed),
        added=added,
        removed=removed,
        preserved=same,
        geometry=details,
    ), supports


def certified_oversize(item, catalog, objects, mapping):
    """A fixed-orientation XY extent cannot fit in the support's enclosing box."""
    mesh = trimesh.load(catalog[item['asset']], force='mesh')
    mesh.apply_transform(
        trimesh.transformations.euler_matrix(1.5708, 0, 0, 'sxyz')
    )
    support = objects[mapping[item['on_instance']]]['mesh']
    return bool(np.any(mesh.extents[:2] > support.extents[:2] + 0.002))
