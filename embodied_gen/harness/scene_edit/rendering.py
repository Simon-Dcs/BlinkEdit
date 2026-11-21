"""Task-independent SAPIEN loading, paired previews and support diagnostics.

Adapted from the prior measured renderer; original collision meshes are used.
"""

import copy
import os
import subprocess
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

import numpy as np
import sapien
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from embodied_gen.utils.simulation import (
    SapienSceneManager,
    capture_frame,
    create_recording_camera,
    load_actor_from_urdf,
    set_ground_base_color,
)

CAMERAS = dict(
    overview=([5, -8, 7], [0, 0, 0.3], 45),
    office=([-4.2, -2.8, 2.8], [-1.7, 0, 0.5], 40),
)


def parse(path, *, geometry_only=False, cached_mesh=None):
    """These frozen fixtures have base-parent fixed joints and no mesh scaling."""
    path = Path(path)
    tree = ET.parse(path)
    transforms = {
        j.find('child').get('link'): j.find('origin')
        for j in tree.findall('joint')
    }
    objects = {}
    for link in tree.findall('link'):
        name = link.get('name')
        if name == 'base':
            continue
        origin = transforms[name]
        xyz = np.fromstring(origin.get('xyz'), sep=' ')
        rpy = np.fromstring(origin.get('rpy'), sep=' ')
        files = [
            path.parent / n.get('filename') for n in link.findall('.//mesh')
        ]
        for file in files:
            if not file.is_file():
                raise ValueError(f'Missing mesh {file}')
        mesh = cached_mesh(name, files[0], xyz, rpy) if cached_mesh else None
        if mesh is None:
            mesh = trimesh.load(
                files[0], force='mesh', skip_materials=geometry_only
            )
            mesh.apply_transform(
                trimesh.transformations.compose_matrix(
                    translate=xyz, angles=rpy
                )
            )
        objects[name] = dict(
            xyz=xyz,
            rpy=rpy,
            mesh=mesh,
            link=link,
            visual=str(files[0].resolve()),
            collision=str(files[1].resolve()) if len(files) > 1 else None,
            asset=files[0].stem,
        )
    return objects


class Renderer:
    """Two persistent scenes, with exact removal of per-request entities."""

    def __init__(self, gpu_index=4):
        self.gpu_index = gpu_index
        self.gpu_uuid = subprocess.check_output(
            [
                'nvidia-smi',
                '-i',
                str(gpu_index),
                '--query-gpu=uuid',
                '--format=csv,noheader',
            ],
            text=True,
        ).strip()
        self.main, self.diagnostic = self.manager(), self.manager()
        self.cameras = {
            k: create_recording_camera(
                self.main, eye, target, (720, 1080), fov, cam_name=k
            )
            for k, (eye, target, fov) in CAMERAS.items()
        }
        self.diag_camera = create_recording_camera(
            self.diagnostic, *CAMERAS['office'][:2], (720, 1080), 40
        )
        self.articulation = None
        self.actors = []

    @staticmethod
    def manager():
        factory = sapien.render.RenderSystem
        sapien.render.RenderSystem = partial(factory, device='cuda:0')
        try:
            manager = SapienSceneManager(
                sim_freq=240, ray_tracing=False, device='cuda:0'
            )
        finally:
            sapien.render.RenderSystem = factory
        set_ground_base_color(manager.scene, [0.88, 0.90, 0.93, 1])
        for entity in manager.scene.entities:
            if entity.name == 'ground':
                entity.set_pose(sapien.Pose(p=[0, 0, -0.081]))
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
            Image.fromarray(capture_frame(self.main.scene, camera)).save(
                directory / f'{prefix}_{name}.png'
            )
        for _ in range(120):
            self.main.scene.step()
        return dict(
            links=len(self.articulation.get_links()),
            collision_shapes=sum(
                len(link.get_collision_shapes())
                for link in self.articulation.get_links()
            ),
        )

    def gpu(self):
        lines = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                '--format=csv',
            ],
            text=True,
        ).splitlines()
        own = [
            line
            for line in lines[1:]
            if line.split(',')[1].strip() == str(os.getpid())
        ]
        assert own and all(self.gpu_uuid in line for line in own), own
        return dict(
            pid=os.getpid(),
            visible=os.environ['CUDA_VISIBLE_DEVICES'],
            rows=own,
            render_device=str(self.main.scene.render_system.device),
        )

    def release(self, objects, added, directory, supports, video=False):
        dm = self.diagnostic
        for actor in self.actors:
            dm.scene.remove_entity(actor)
        self.actors.clear()
        dynamics = {}
        adapt = directory / 'validation_adapters'
        adapt.mkdir(exist_ok=True)
        for name, info in objects.items():
            single = ET.Element('robot', name=name)
            link = copy.deepcopy(info['link'])
            single.append(link)
            for kind in ['visual', 'collision']:
                part = link.find(kind)
                assert part is not None, f'Missing {kind} on {name}'
                ET.SubElement(
                    part,
                    'origin',
                    xyz='0 0 0',
                    rpy=' '.join(map(str, info['rpy'])),
                )
                part.find('geometry/mesh').set('filename', info[kind])
            gaz = ET.SubElement(link.find('collision'), 'gazebo')
            ET.SubElement(gaz, 'mu1').text = '.6'
            ET.SubElement(gaz, 'mu2').text = '.5'
            path = adapt / f'{name}.urdf'
            ET.ElementTree(single).write(path)
            actor = load_actor_from_urdf(
                dm.scene,
                str(path),
                pose=sapien.Pose(p=info['xyz']),
                use_static=name not in added,
            )
            self.actors.append(actor)
            for component in actor.components:
                if isinstance(component, sapien.physx.PhysxRigidBodyComponent):
                    for shape in component.collision_shapes:
                        shape.set_contact_offset(0.001)
                        shape.set_rest_offset(0)
            if name in added:
                dynamics[name] = actor
        initial = {
            name: (actor.pose.p.copy(), actor.pose.q.copy())
            for name, actor in dynamics.items()
        }
        trajectories = {name: [] for name in dynamics}
        frames = []
        for step in range(480):
            dm.scene.step()
            if step % 8 == 0:
                if video:
                    frames.append(capture_frame(dm.scene, self.diag_camera))
                for name, actor in dynamics.items():
                    trajectories[name].append(actor.pose.p.tolist())
        Image.fromarray(capture_frame(dm.scene, self.diag_camera)).save(
            directory / 'release_final.png'
        )
        results = {}
        for name, actor in dynamics.items():
            table_bounds = objects[supports[name]]['mesh'].bounds
            xyz, q = initial[name]
            translation = float(np.linalg.norm(actor.pose.p - xyz))
            angle = float(
                np.degrees(
                    2 * np.arccos(np.clip(abs(np.dot(q, actor.pose.q)), 0, 1))
                )
            )
            # The diagnostic orientation stays in visual/collision local pose.
            rot = Rotation.from_quat(np.roll(actor.pose.q, -1)).as_matrix()
            local = objects[name]['mesh'].vertices - xyz
            bounds = (
                (local @ rot.T + actor.pose.p).min(0),
                (local @ rot.T + actor.pose.p).max(0),
            )
            supported = (
                bounds[0][2] >= table_bounds[1][2] - 0.02
                and bounds[0][2] <= table_bounds[1][2] + 0.03
                and np.all(bounds[0][:2] >= table_bounds[0, :2] - 0.002)
                and np.all(bounds[1][:2] <= table_bounds[1, :2] + 0.002)
            )
            results[name] = dict(
                initial_xyz=xyz.tolist(),
                final_xyz=actor.pose.p.tolist(),
                translation_m=translation,
                rotation_deg=angle,
                still_supported=bool(supported),
                stable=bool(translation <= 0.02 and angle <= 5 and supported),
                trajectory=trajectories[name],
            )
        return results, frames
