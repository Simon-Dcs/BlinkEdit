"""Create unedited controlled input rooms from official demo assets.

No target edits occur here. Joint transforms encode the source URDF visual
orientation because the native collector reads joint transforms, not visuals.
"""
import copy
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

start = time.perf_counter()
root = Path('outputs/efficiency/latest_run.txt').read_text().strip()
root = Path(root)
dest = root/'fixtures/two_rooms'
dest.mkdir()
robot = ET.Element('robot', name='controlled_two_room_input')
ET.SubElement(robot, 'link', name='base')
manifest = []


def add(name, meshfile, collisionfile, xyz, rpy):
    link = ET.SubElement(robot, 'link', name=name)
    for kind, path in [('visual', meshfile), ('collision', collisionfile)]:
        tag = ET.SubElement(link, kind)
        geo = ET.SubElement(tag, 'geometry')
        ET.SubElement(geo, 'mesh', filename=str(path))
    joint = ET.SubElement(robot, 'joint', name='joint_'+name, type='fixed')
    ET.SubElement(joint, 'parent', link='base')
    ET.SubElement(joint, 'child', link=name)
    ET.SubElement(joint, 'origin', xyz=' '.join(map(str, xyz)), rpy=' '.join(map(str, rpy)))


for room, x in [('office', -1.7), ('dining', 1.7)]:
    floor = trimesh.creation.box([3.4, 3.4, .08])
    floor.visual.vertex_colors = [210, 220, 230, 255] if room == 'office' else [230, 220, 200, 255]
    fname = f'{room}_floor.obj'
    floor.export(dest/fname)
    add(f'{room}_0_floor', fname, fname, [x, 0, -.04], [0, 0, 0])
    for asset, xy in [('table', [x, 0]), ('lamp', [x, 1.25])]:
        key = f'{asset}_{room}_001'
        shutil.copytree(root/'source'/asset/'mesh', dest/key)
        src = ET.parse(root/'source'/asset/f'{asset}.urdf').getroot()
        mesh = trimesh.load(dest/key/f'{asset}.obj', force='mesh')
        rpy = np.fromstring(src.find('.//visual/origin').get('rpy'), sep=' ')
        rotated = mesh.vertices@Rotation.from_euler('xyz', rpy).as_matrix().T
        z = -float(rotated[:, 2].min())
        add(key, f'{key}/{asset}.obj', f'{key}/{asset}_collision.obj', [*xy, z], rpy)
        manifest.append(dict(instance=key, source=str(root/'source'/asset), xyz=[*xy,z], rpy=rpy.tolist(), bounds=(np.array([rotated.min(0),rotated.max(0)])+[xy[0],xy[1],z]).tolist(), faces=len(mesh.faces)))
ET.indent(robot)
ET.ElementTree(robot).write(dest/'scene.urdf', encoding='utf-8', xml_declaration=True)
# Isolated asset directory: prevents native copytree from copying unrelated meshes.
(root/'source/sofa/mesh').mkdir(parents=True)
shutil.copy2('apps/assets/example_texture/meshes/sofa.obj', root/'source/sofa/mesh/sofa.obj')
(root/'setup/fixture.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start, description='Two adjacent 3.4 x 3.4 m input rooms; official demo table and lamp in each; no generation claim.', instances=manifest),indent=2))
print(json.dumps(manifest, indent=2))
