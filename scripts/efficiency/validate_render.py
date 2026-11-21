"""Validate actual URDF outputs and render them on physical GPU 7.

Validation is independent of editing. It never changes scene meshes or poses.
The optional release diagnostic loads original collision meshes as separate
actors, to test support under gravity rather than treating fixed joints as proof.
"""
import argparse
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

START = time.perf_counter()
import numpy as np
import sapien
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPoint
from embodied_gen.utils.simulation import (SapienSceneManager, capture_frame,
    create_recording_camera, set_ground_base_color, load_actor_from_urdf)

p = argparse.ArgumentParser()
p.add_argument('--case', required=True, help='baseline or run directory basename')
p.add_argument('--release', action='store_true')
args = p.parse_args()
root = Path(Path('outputs/efficiency/latest_run.txt').read_text().strip())
baseline = args.case == 'baseline'
d = root/'comparison/baseline' if baseline else root/'runs'/args.case
d.mkdir(parents=True, exist_ok=True)
urdf = root/'fixtures/two_rooms/scene.urdf' if baseline else d/'scene/scene_updated.urdf'
if not urdf.exists():
    urdf = d/'scene/scene.urdf'
timings = {'imports': time.perf_counter()-START}
records = {}
assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'


def save(name, value):
    (d/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n')


def parse(path):
    tree = ET.parse(path)
    transforms = {j.find('child').get('link'): j.find('origin') for j in tree.findall('joint')}
    objects = {}
    for link in tree.findall('link'):
        name = link.get('name')
        if name == 'base':
            continue
        origin = transforms[name]
        xyz = np.fromstring(origin.get('xyz'), sep=' ')
        rpy = np.fromstring(origin.get('rpy'), sep=' ')
        paths = [path.parent/n.get('filename') for n in link.findall('.//mesh')]
        for file in paths:
            assert file.exists(), file
        mesh = trimesh.load(paths[0], force='mesh')
        mesh.apply_transform(trimesh.transformations.compose_matrix(translate=xyz, angles=rpy))
        objects[name] = dict(xyz=xyz, rpy=rpy, mesh=mesh, link=link,
                             visual=str(paths[0]), collision=str(paths[1]) if len(paths)>1 else None)
    return objects


t = time.perf_counter()
objects = parse(urdf)
before = parse(root/'fixtures/two_rooms/scene.urdf')
added = sorted(set(objects)-set(before))
removed = sorted(set(before)-set(objects))
preserved = all(np.allclose(objects[k]['xyz'], before[k]['xyz']) and
                np.allclose(objects[k]['rpy'], before[k]['rpy']) for k in set(objects)&set(before))
table = objects['table_office_001']['mesh']
table_hull = MultiPoint(table.vertices[:, :2]).convex_hull
for name in added:
    mesh = objects[name]['mesh']
    hull = MultiPoint(mesh.vertices[:, :2]).convex_hull
    records[name] = dict(bounds=mesh.bounds.tolist(), support_gap_to_table_bbox_m=float(mesh.bounds[0,2]-table.bounds[1,2]),
                         asset_type=Path(objects[name]['visual']).stem,
                         xy_within_table_hull=table_hull.buffer(.002).covers(hull),
                         collision_present=objects[name]['collision'] is not None)
overlaps = []
for i, a in enumerate(added):
    for b in added[i+1:]:
        ha = MultiPoint(objects[a]['mesh'].vertices[:, :2]).convex_hull
        hb = MultiPoint(objects[b]['mesh'].vertices[:, :2]).convex_hull
        overlaps.append(dict(a=a,b=b,conservative_xy_overlap_m2=ha.intersection(hb).area))
timings['geometry_inventory'] = time.perf_counter()-t
save('validation.json', dict(input=str(urdf), added=added, removed=removed, unchanged_poses=preserved,
                              total_instances=len(objects), assets=records, pairwise=overlaps,
                              note='Convex XY hull is a conservative diagnostic, not exact 3D collision detection.'))


def manager():
    factory = sapien.render.RenderSystem
    sapien.render.RenderSystem = partial(factory, device='cuda:0')
    try:
        m = SapienSceneManager(sim_freq=240, ray_tracing=False, device='cuda:0')
    finally:
        sapien.render.RenderSystem = factory
    set_ground_base_color(m.scene, [.88,.90,.93,1])
    # Input floors already occupy z=0. Move the auxiliary infinite ground below
    # them to prevent z-fighting without changing any authored scene object.
    for entity in m.scene.entities:
        if entity.name == 'ground':
            entity.set_pose(sapien.Pose(p=[0,0,-.081]))
    return m


t = time.perf_counter()
m = manager()
loader = m.scene.create_urdf_loader()
loader.fix_root_link = True
loader.load_multiple_collisions_from_file = True
articulation = loader.load(str(urdf))
assert articulation is not None
# SAPIEN initializes fixed-joint world transforms at the first physics step.
m.scene.step()
timings['sapien_load'] = time.perf_counter()-t
gpu_output = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
gpu_lines = gpu_output.splitlines()
own_gpu_lines = [line for line in gpu_lines[1:] if line.split(',')[1].strip() == str(os.getpid())]
assert own_gpu_lines and all('GPU-63a185b7-8463-446a-808b-10425a0e9800' in line for line in own_gpu_lines), own_gpu_lines
save('gpu.json', dict(render_device=str(m.scene.render_system.device), process_pid=os.getpid(),
                     visible=os.environ['CUDA_VISIBLE_DEVICES'],
                     processes='\n'.join([gpu_lines[0],*own_gpu_lines])))
t = time.perf_counter()
cameras = dict(overview=([5,-8,7],[0,0,.3],45), office=([-4.2,-2.8,2.8],[-1.7,0,.5],40))
for key, (eye,target,fov) in cameras.items():
    cam = create_recording_camera(m, eye, target, (720,1080), fov, cam_name=key)
    Image.fromarray(capture_frame(m.scene, cam)).save(d/f'{key}.png')
for _ in range(120):
    m.scene.step()
timings['two_views_and_fixed_scene_steps'] = time.perf_counter()-t
save('sapien_load.json', dict(links=len(articulation.get_links()), steps=120,
                             collision_shapes=sum(len(link.get_collision_shapes()) for link in articulation.get_links()),
                             note='Original scene uses fixed joints; stepping it alone is not a stability test.'))

if args.release and added:
    t = time.perf_counter()
    dm = manager()
    dynamics = {}
    adapt = d/'validation_adapters'
    adapt.mkdir(exist_ok=True)
    for name, info in objects.items():
        # Single-link wrappers only for the native actor loader; identical mesh
        # geometry and authored pose. Diagnostic friction is explicit; density
        # remains the native actor loader default, not a copied source mass.
        single = ET.Element('robot',name=name)
        link = ET.fromstring(ET.tostring(info['link']))
        single.append(link)
        for kind in ['visual','collision']:
            part = link.find(kind)
            ET.SubElement(part,'origin',xyz='0 0 0',rpy=' '.join(map(str,info['rpy'])))
            part.find('geometry/mesh').set('filename',info[kind])
        gaz = ET.SubElement(link.find('collision'),'gazebo')
        ET.SubElement(gaz,'mu1').text='.6'
        ET.SubElement(gaz,'mu2').text='.5'
        path = adapt/f'{name}.urdf'
        ET.ElementTree(single).write(path)
        actor = load_actor_from_urdf(dm.scene, str(path), pose=sapien.Pose(p=info['xyz']), use_static=name not in added)
        # Same 1 mm contact offset for all diagnostic actors.
        for component in actor.components:
            if isinstance(component,sapien.physx.PhysxRigidBodyComponent):
                for shape in component.collision_shapes:
                    shape.set_contact_offset(.001)
                    shape.set_rest_offset(0)
        if name in added:
            dynamics[name] = actor
    initial = {name: actor.pose.p.copy() for name,actor in dynamics.items()}
    trajectories = {name: [] for name in dynamics}
    cam = create_recording_camera(dm,*cameras['office'][:2],(720,1080),40)
    import imageio.v2 as imageio
    frames = []
    for step in range(480):
        dm.scene.step()
        if step%8==0:
            frames.append(capture_frame(dm.scene,cam))
            for name,actor in dynamics.items():
                trajectories[name].append(actor.pose.p.tolist())
    imageio.mimwrite(d/'release.mp4',frames,fps=30,macro_block_size=1)
    Image.fromarray(frames[-1]).save(d/'release_final.png')
    save('release.json', dict(seconds_simulated=2, diagnostics={name:dict(initial_xyz=initial[name].tolist(),
        final_xyz=actor.pose.p.tolist(),translation_m=float(np.linalg.norm(actor.pose.p-initial[name])),
        final_q=actor.pose.q.tolist(),trajectory=trajectories[name]) for name,actor in dynamics.items()},
        adaptation='Separate actors, source collision meshes, friction .6/.5, native loader density default, 1mm contact offset; diagnostic only, original URDF unchanged.'))
    timings['release_validation_with_video'] = time.perf_counter()-t
timings['total'] = time.perf_counter()-START
save('validation_timings.json',timings)
print(args.case, timings, flush=True)
