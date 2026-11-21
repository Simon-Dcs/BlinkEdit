"""Edit embedded glTF without re-encoding untouched mesh, UV or image data."""

import copy
import io
import itertools
import json
import struct
import uuid
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def header(path):
    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        n, typ = struct.unpack('<I4s', f.read(8))
        if (magic, version, typ) != (b'glTF', 2, b'JSON') or length != Path(
            path
        ).stat().st_size:
            raise ValueError('Invalid GLB container')
        g = json.loads(f.read(n))
        size, kind = struct.unpack('<I4s', f.read(8))
        if (
            kind != b'BIN\0'
            or size + f.tell() != length
            or len(g['buffers']) != 1
            or any('uri' in x for x in [*g['buffers'], *g.get('images', [])])
        ):
            raise ValueError('Expected a self-contained binary GLB')
        return g, f.tell()


def write(path, g, blob):
    blob = bytes(blob)
    g['buffers'][0]['byteLength'] = len(blob)
    j = json.dumps(g, separators=(',', ':'), allow_nan=False).encode()
    j += b' ' * (-len(j) % 4)
    blob += b'\0' * (-len(blob) % 4)
    with open(path, 'wb') as f:
        f.write(
            struct.pack(
                '<4sIII4s',
                b'glTF',
                2,
                28 + len(j) + len(blob),
                len(j),
                b'JSON',
            )
        )
        f.write(j)
        f.write(struct.pack('<I4s', len(blob), b'BIN\0'))
        f.write(blob)


def surfaces(g, mesh):
    materials = []
    for primitive in mesh['primitives']:
        m = (
            g['materials'][primitive['material']]
            if 'material' in primitive
            else {}
        )
        pbr = m.get('pbrMetallicRoughness', {})
        materials.append(
            dict(
                name=m.get('name'),
                factor=pbr.get('baseColorFactor', [1] * 4),
                texture=pbr.get('baseColorTexture', {}).get('index'),
            )
        )
    return materials


def geometry(g, i, matrix):
    node = g['nodes'][i]
    mesh = g['meshes'][node['mesh']]
    bounds = []
    count = 0
    for p in mesh['primitives']:
        acc = g['accessors'][p['attributes']['POSITION']]
        corners = np.array(
            list(itertools.product(*zip(acc['min'], acc['max'])))
        )
        bounds.extend(corners @ matrix[:3, :3].T + matrix[:3, 3])
        count += acc['count']
    return dict(
        name=node.get('name'),
        bounds=np.round(
            [np.min(bounds, axis=0), np.max(bounds, axis=0)], 6
        ).tolist(),
        vertices=count,
        materials=surfaces(g, mesh),
    )


def observe(g, previous=None, changed=(), painted=()):
    # Only trusted operations below may authorize reuse. An external change has no such delta.
    nodes = {}
    seen = set()
    n_calc = 0

    def visit(i, parent, world):
        nonlocal n_calc
        if i in seen:
            raise ValueError('Non-tree node graph')
        seen.add(i)
        node = g['nodes'][i]
        if 'matrix' in node:
            local = np.array(node['matrix']).reshape(4, 4).T
        else:
            local = np.eye(4)
            local[:3, 3] = node.get('translation', [0] * 3)
            local[:3, :3] = Rotation.from_quat(
                node.get('rotation', [0, 0, 0, 1])
            ).as_matrix() @ np.diag(node.get('scale', [1] * 3))
        matrix = world @ local
        if 'mesh' in node:
            k = str(i)
            if (
                previous is not None
                and i not in changed
                and k in previous['nodes']
            ):
                nodes[k] = previous['nodes'][k]
                if i in painted:
                    nodes[k] = dict(
                        nodes[k],
                        materials=surfaces(g, g['meshes'][node['mesh']]),
                    )
            else:
                nodes[k] = dict(
                    parent=parent,
                    world=matrix.tolist(),
                    **geometry(g, i, matrix),
                )
                n_calc += 1
            nodes[k] = dict(
                nodes[k], stable_id=node.get('extras', {}).get('scene_edit_id')
            )
        for child in node.get('children', []):
            visit(child, i, matrix)

    for i in g['scenes'][g.get('scene', 0)]['nodes']:
        visit(i, None, np.eye(4))
    if not nodes:
        raise ValueError('Empty scene')
    ids = [n['stable_id'] for n in nodes.values() if n['stable_id']]
    for n in nodes.values():
        if n['stable_id'] and ids.count(n['stable_id']) != 1:
            n['stable_id'] = None
    return dict(
        schema='embedded-glb-v2', axes='glTF world, +Y up, metres', nodes=nodes
    ), n_calc


def edit(
    source, destination, ops, before, incremental, *, initialize_ids=False, trusted_restores=None
):
    g, offset = header(source)
    with open(source, 'rb') as f:
        f.seek(offset)
        blob = bytearray(f.read())
    if initialize_ids:
        for node in g['nodes']:
            node.setdefault('extras', {}).setdefault(
                'scene_edit_id', uuid.uuid4().hex
            )
    changed = set()
    painted = set()

    def accessor(array, dtype, kind, component, target):
        a = np.ascontiguousarray(array, dtype=dtype)
        blob.extend(b'\0' * (-len(blob) % 4))
        view = len(g['bufferViews'])
        g['bufferViews'].append(
            dict(
                buffer=0,
                byteOffset=len(blob),
                byteLength=a.nbytes,
                target=target,
            )
        )
        blob.extend(a.tobytes())
        idx = len(g['accessors'])
        data = dict(
            bufferView=view, componentType=component, count=len(a), type=kind
        )
        if kind == 'VEC3':
            data.update(min=a.min(axis=0).tolist(), max=a.max(axis=0).tolist())
        g['accessors'].append(data)
        return idx

    def material(op, original=None):
        c = np.array(op['color'], dtype=float)
        if (
            c.shape != (3,)
            or not np.isfinite(c).all()
            or np.any((c < 0) | (c > 1))
        ):
            raise ValueError('RGB out of range')
        mat = copy.deepcopy(original) if original is not None else {}
        mat['name'] = op.get('name', 'edited')
        pbr = mat.setdefault('pbrMetallicRoughness', {})
        pbr.update(
            baseColorFactor=[*c, 1.0],
            metallicFactor=float(
                op.get('metallic', pbr.get('metallicFactor', 0))
            ),
            roughnessFactor=float(
                op.get('roughness', pbr.get('roughnessFactor', 0.65))
            ),
        )
        if not op.get('keep_texture', False):
            pbr.pop('baseColorTexture', None)
        if 'pixels' in op:
            from PIL import Image

            pixels = np.asarray(op['pixels'])
            if (
                pixels.ndim != 3
                or pixels.shape[2] != 3
                or not (0 < max(pixels.shape[:2]) <= 1024)
                or pixels.min() < 0
                or pixels.max() > 255
            ):
                raise ValueError(
                    'Expected an RGB texture of at most 1024 x 1024'
                )
            stream = io.BytesIO()
            Image.fromarray(pixels.astype('uint8')).save(stream, format='PNG')
            raw = stream.getvalue()
            blob.extend(b'\0' * (-len(blob) % 4))
            vi = len(g['bufferViews'])
            g['bufferViews'].append(
                dict(buffer=0, byteOffset=len(blob), byteLength=len(raw))
            )
            blob.extend(raw)
            im = len(g.setdefault('images', []))
            g['images'].append(dict(bufferView=vi, mimeType='image/png'))
            tx = len(g.setdefault('textures', []))
            g['textures'].append(dict(source=im))
            pbr['baseColorTexture'] = {'index': tx}
        g.setdefault('materials', []).append(mat)
        return len(g['materials']) - 1

    for op in ops:
        action = op['op']
        if action in ('remove', 'paint'):
            i = int(op['id'])
            if str(i) not in before['nodes'] or 'mesh' not in g['nodes'][i]:
                raise ValueError('Missing target')
            node = g['nodes'][i]
            (changed if action == 'remove' else painted).add(i)
            if action == 'remove':
                node.pop('mesh')  # retain children and their transforms
            else:
                mesh = g['meshes'][node['mesh']].copy()
                mesh['primitives'] = [
                    dict(
                        p,
                        material=material(
                            op,
                            g['materials'][p['material']]
                            if 'material' in p
                            else None,
                        ),
                    )
                    for p in mesh['primitives']
                ]
                node['mesh'] = len(g['meshes'])
                g['meshes'].append(mesh)
        elif action == 'restore' and trusted_restores:
            i = int(op['id'])
            mesh = int(op['mesh'])
            if trusted_restores.get(i) != mesh or 'mesh' in g['nodes'][i]:
                raise ValueError('Invalid retired part')
            g['nodes'][i]['mesh'] = mesh
            changed.add(i)
        elif action == 'add':
            shape = op['shape']
            if shape == 'mesh':
                mesh = trimesh.Trimesh(
                    vertices=op['vertices'], faces=op['faces'], process=False
                )
            else:
                make = {
                    'box': trimesh.creation.box,
                    'sphere': lambda: trimesh.creation.icosphere(
                        subdivisions=2
                    ),
                    'cylinder': lambda: trimesh.creation.cylinder(
                        radius=1, height=2, sections=32
                    ),
                    'cone': lambda: trimesh.creation.cone(
                        radius=1, height=2, sections=32
                    ),
                }
                mesh = make[shape]()
                if shape in ('cone', 'cylinder'):
                    mesh.apply_transform(
                        trimesh.transformations.rotation_matrix(
                            -np.pi / 2, [1, 0, 0]
                        )
                    )
                mesh.apply_translation(-mesh.bounds.mean(axis=0))
                mesh.apply_scale(np.array(op['size']) / mesh.extents)
                mesh.vertices = (
                    Rotation.from_euler(
                        'xyz', op.get('rotation', [0] * 3), degrees=True
                    ).apply(mesh.vertices)
                    + op['center']
                )
            if (
                len(mesh.faces) == 0
                or not np.isfinite(mesh.vertices).all()
                or mesh.faces.min() < 0
                or mesh.faces.max() >= len(mesh.vertices)
            ):
                raise ValueError('Invalid generated geometry')
            parent = int(op['attach_to']) if 'attach_to' in op else None
            if parent is not None:
                if (
                    str(parent) not in before['nodes']
                    or 'mesh' not in g['nodes'][parent]
                ):
                    raise ValueError('Missing attachment parent')
                mesh.apply_transform(
                    np.linalg.inv(
                        np.asarray(before['nodes'][str(parent)]['world'])
                    )
                )
            pos = accessor(mesh.vertices, '<f4', 'VEC3', 5126, 34962)
            ind = accessor(
                mesh.faces.reshape(-1), '<u4', 'SCALAR', 5125, 34963
            )
            mid = len(g['meshes'])
            g['meshes'].append(
                dict(
                    primitives=[
                        dict(
                            attributes=dict(POSITION=pos),
                            indices=ind,
                            material=material(op),
                        )
                    ]
                )
            )
            i = len(g['nodes'])
            g['nodes'].append(
                dict(
                    name=str(op['name']),
                    mesh=mid,
                    extras=dict(scene_edit_id=uuid.uuid4().hex),
                )
            )
            (
                g['scenes'][g.get('scene', 0)]['nodes']
                if parent is None
                else g['nodes'][parent].setdefault('children', [])
            ).append(i)
            changed.add(i)
        else:
            raise ValueError(f'Unknown operation: {action}')
    write(destination, g, blob)
    actual, _ = header(destination)
    return observe(actual, before if incremental else None, changed, painted)
