"""Independent, task-aware acceptance; never supplies targets to the planner."""
import copy
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

import numpy as np
import sapien
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPoint
from embodied_gen.utils.simulation import (SapienSceneManager, capture_frame,
    create_recording_camera, load_actor_from_urdf, set_ground_base_color)
from common import save

GPU_UUID = 'GPU-63a185b7-8463-446a-808b-10425a0e9800'
CAMERAS = dict(overview=([5,-8,7],[0,0,.3],45), office=([-4.2,-2.8,2.8],[-1.7,0,.5],40))


def parse(path):
    """These frozen fixtures have base-parent fixed joints and no mesh scaling."""
    path = Path(path)
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
        files = [path.parent/n.get('filename') for n in link.findall('.//mesh')]
        for file in files:
            if not file.is_file():
                raise ValueError(f'Missing mesh {file}')
        mesh = trimesh.load(files[0], force='mesh')
        mesh.apply_transform(trimesh.transformations.compose_matrix(translate=xyz, angles=rpy))
        objects[name] = dict(xyz=xyz, rpy=rpy, mesh=mesh, link=link, visual=str(files[0].resolve()),
                             collision=str(files[1].resolve()) if len(files)>1 else None,
                             asset=files[0].stem)
    return objects


class Renderer:
    """Two persistent scenes, with exact removal of per-request entities."""
    def __init__(self):
        self.main, self.diagnostic = self.manager(), self.manager()
        self.cameras = {k: create_recording_camera(self.main, eye, target, (720,1080), fov, cam_name=k)
                        for k,(eye,target,fov) in CAMERAS.items()}
        self.diag_camera = create_recording_camera(self.diagnostic, *CAMERAS['office'][:2], (720,1080), 40)
        self.articulation = None
        self.actors = []

    @staticmethod
    def manager():
        factory = sapien.render.RenderSystem
        sapien.render.RenderSystem = partial(factory, device='cuda:0')
        try:
            manager = SapienSceneManager(sim_freq=240, ray_tracing=False, device='cuda:0')
        finally:
            sapien.render.RenderSystem = factory
        set_ground_base_color(manager.scene, [.88,.90,.93,1])
        for entity in manager.scene.entities:
            if entity.name == 'ground':
                entity.set_pose(sapien.Pose(p=[0,0,-.081]))
        return manager

    def reset(self):
        if self.articulation is not None:
            self.main.scene.remove_articulation(self.articulation)
            self.articulation = None
        for actor in self.actors:
            self.diagnostic.scene.remove_entity(actor)
        self.actors.clear()

    def preview(self, path, directory, prefix):
        if self.articulation is not None:
            self.main.scene.remove_articulation(self.articulation)
        loader = self.main.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        self.articulation = loader.load(str(path))
        assert self.articulation is not None
        self.main.scene.step()
        for name, camera in self.cameras.items():
            Image.fromarray(capture_frame(self.main.scene, camera)).save(directory/f'{prefix}_{name}.png')
        for _ in range(120):
            self.main.scene.step()
        return dict(links=len(self.articulation.get_links()),
                    collision_shapes=sum(len(l.get_collision_shapes()) for l in self.articulation.get_links()))

    def gpu(self):
        lines = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                         '--format=csv'],text=True).splitlines()
        own = [l for l in lines[1:] if l.split(',')[1].strip() == str(os.getpid())]
        assert own and all(GPU_UUID in l for l in own), own
        return dict(pid=os.getpid(), visible=os.environ['CUDA_VISIBLE_DEVICES'], rows=own,
                    render_device=str(self.main.scene.render_system.device))

    def release(self, objects, added, directory, video=False):
        dm = self.diagnostic
        for actor in self.actors:
            dm.scene.remove_entity(actor)
        self.actors.clear()
        dynamics = {}
        adapt = directory/'validation_adapters'
        adapt.mkdir(exist_ok=True)
        for name, info in objects.items():
            single = ET.Element('robot',name=name)
            link = copy.deepcopy(info['link'])
            single.append(link)
            for kind in ['visual','collision']:
                part = link.find(kind)
                assert part is not None, f'Missing {kind} on {name}'
                ET.SubElement(part,'origin',xyz='0 0 0',rpy=' '.join(map(str,info['rpy'])))
                part.find('geometry/mesh').set('filename',info[kind])
            gaz = ET.SubElement(link.find('collision'),'gazebo')
            ET.SubElement(gaz,'mu1').text='.6'
            ET.SubElement(gaz,'mu2').text='.5'
            path = adapt/f'{name}.urdf'
            ET.ElementTree(single).write(path)
            actor = load_actor_from_urdf(dm.scene, str(path), pose=sapien.Pose(p=info['xyz']), use_static=name not in added)
            self.actors.append(actor)
            for component in actor.components:
                if isinstance(component, sapien.physx.PhysxRigidBodyComponent):
                    for shape in component.collision_shapes:
                        shape.set_contact_offset(.001)
                        shape.set_rest_offset(0)
            if name in added:
                dynamics[name] = actor
        initial = {name: (actor.pose.p.copy(), actor.pose.q.copy()) for name,actor in dynamics.items()}
        trajectories = {name: [] for name in dynamics}
        frames = []
        for step in range(480):
            dm.scene.step()
            if step%8==0:
                if video:
                    frames.append(capture_frame(dm.scene,self.diag_camera))
                for name,actor in dynamics.items():
                    trajectories[name].append(actor.pose.p.tolist())
        Image.fromarray(capture_frame(dm.scene,self.diag_camera)).save(directory/'release_final.png')
        results = {}
        table_bounds = objects['table_office_001']['mesh'].bounds
        for name, actor in dynamics.items():
            xyz, q = initial[name]
            translation = float(np.linalg.norm(actor.pose.p-xyz))
            angle = float(np.degrees(2*np.arccos(np.clip(abs(np.dot(q, actor.pose.q)), 0, 1))))
            # The diagnostic orientation stays in visual/collision local pose.
            rot = Rotation.from_quat(np.roll(actor.pose.q, -1)).as_matrix()
            local = objects[name]['mesh'].vertices-xyz
            bounds = (local@rot.T+actor.pose.p).min(0), (local@rot.T+actor.pose.p).max(0)
            supported = (bounds[0][2] >= table_bounds[1][2]-.02 and
                         bounds[0][2] <= table_bounds[1][2]+.03 and
                         np.all(bounds[0][:2] >= table_bounds[0,:2]-.002) and
                         np.all(bounds[1][:2] <= table_bounds[1,:2]+.002))
            results[name] = dict(initial_xyz=xyz.tolist(), final_xyz=actor.pose.p.tolist(),
                translation_m=translation, rotation_deg=angle, still_supported=bool(supported),
                stable=bool(translation <= .02 and angle <= 5 and supported), trajectory=trajectories[name])
        return results, frames


def validate(task, before_path, after_path, status, renderer, directory, meter, video=False):
    with meter.span('geometry_validation'):
        before, objects = parse(before_path), parse(after_path)
        added, removed = sorted(set(objects)-set(before)), sorted(set(before)-set(objects))
        kept = set(objects)&set(before)
        preserved = all(np.array_equal(objects[k]['xyz'], before[k]['xyz']) and
                        np.array_equal(objects[k]['rpy'], before[k]['rpy']) and
                        np.array_equal(objects[k]['mesh'].vertices, before[k]['mesh'].vertices) and
                        Path(objects[k]['collision']).read_bytes() == Path(before[k]['collision']).read_bytes()
                        for k in kept)
        expectations = {
            'T1': ([], ['lamp_office_001'], 'execute'),
            'T2': (['vase'], [], 'execute'),
            'T3': (['lamp','vase','book','mug'], [], 'execute'),
            'T4': (['lamp','vase','book','mug'], ['lamp_office_001'], 'execute'),
            'C1': (['mug'], [], 'execute'), 'C2': (['book'], [], 'execute'),
            'C3': ([], [k for k,v in before.items() if v['asset']=='mug'], 'execute'),
            'C4': ([], [], 'noop'), 'B1': ([], [], 'impossible'),
            'B2': ([], [], 'noop'), 'B3': ([], [], 'clarify'),
        }
        exp_add, exp_remove, exp_status = expectations[task]
        task_ok = (sorted(objects[k]['asset'] for k in added) == sorted(exp_add) and
                   removed == sorted(exp_remove) and status == exp_status and preserved)
        table = objects['table_office_001']['mesh']
        table_hull = MultiPoint(table.vertices[:,:2]).convex_hull
        # Include earlier inserted objects when checking inter-object overlap.
        tabletop = [k for k,v in objects.items() if v['asset'] in ['mug','book','vase'] or
                    (v['asset']=='lamp' and k not in ['lamp_office_001','lamp_dining_001'])]
        hulls = {k: MultiPoint(objects[k]['mesh'].vertices[:,:2]).convex_hull for k in tabletop}
        assets = {}
        for name in tabletop:
            mesh = objects[name]['mesh']
            gap = float(mesh.bounds[0,2]-table.bounds[1,2])
            assets[name] = dict(asset=objects[name]['asset'], bounds=mesh.bounds.tolist(), support_gap_m=gap,
                xy_within_table=bool(table_hull.buffer(.002).covers(hulls[name])),
                collision_present=objects[name]['collision'] is not None)
        overlaps = [dict(a=a,b=b,area_m2=float(hulls[a].intersection(hulls[b]).area))
                    for i,a in enumerate(tabletop) for b in tabletop[i+1:]]
        geometry_ok = (all(v['xy_within_table'] and -.002 <= v['support_gap_m'] <= .03 and v['collision_present']
                           for v in assets.values()) and all(v['area_m2'] <= 1e-5 for v in overlaps))
    with meter.span('sapien_load_and_two_views_before_after'):
        before_load = renderer.preview(before_path, directory, 'before')
        after_load = renderer.preview(after_path, directory, 'after')
        gpu = renderer.gpu()
        save(directory/'gpu.json', gpu)
    release, frames = {}, []
    if added:
        with meter.span('support_diagnostic'):
            release, frames = renderer.release(objects, added, directory, video=video)
    stable = all(r['stable'] for r in release.values())
    result = dict(task=task, status=status, added=added, removed=removed, unchanged_poses_and_meshes=preserved,
        task_ok=task_ok, geometry_ok=geometry_ok, support_ok=stable, accepted=bool(task_ok and geometry_ok and stable),
        assets=assets, overlaps=overlaps, release=release, before_load=before_load, after_load=after_load,
        note='Conservative XY hull check, not exact 3D collision proof. Release: 2s, 240Hz, source collision meshes, friction .6/.5, native density, 1mm contact offset; <=2cm and <=5deg, support retained.')
    save(directory/'validation.json', result)
    return result, frames
