"""Blender entry: observe named level bounds before normalization; use the unmodified official renderer."""

import importlib.util
import json
import re
from pathlib import Path

import bpy
from mathutils import Vector

spec = importlib.util.spec_from_file_location('official', '/render.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
original = official.execute_scene_script


def observed(path):
    original(path)
    if Path(path).name != 'modified.py':
        return
    bpy.context.view_layer.update()
    roles = {'front': [], 'middle': [], 'rear': []}
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        words = set(
            re.findall(
                '[a-z]+', re.sub('([a-z])([A-Z])', r'\1 \2', obj.name).lower()
            )
        )
        if not words.intersection(
            {
                'shelf',
                'shelving',
                'level',
                'platform',
                'deck',
                'tier',
                'surface',
            }
        ):
            continue
        if words.intersection(
            {
                'leg',
                'rail',
                'frame',
                'wall',
                'hole',
                'support',
                'pin',
                'side',
                'screw',
                'rib',
                'post',
                'ramp',
            }
        ):
            continue
        hits = [
            role
            for role, aliases in [
                ('front', {'front'}),
                ('middle', {'middle', 'mid', 'central'}),
                ('rear', {'back', 'rear'}),
            ]
            if words & aliases
        ]
        if len(hits) != 1:
            continue
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        z = [(ev.matrix_world @ Vector(v)).z for v in ev.bound_box]
        roles[hits[0]].append(
            {'name': obj.name, 'z_bottom': min(z), 'z_top': max(z)}
        )
    result = {'role_counts': {k: len(v) for k, v in roles.items()}}
    if all(len(v) == 1 for v in roles.values()):
        result['unique_surfaces'] = {k: v[0] for k, v in roles.items()}
    Path('/work/levels.json').write_text(json.dumps(result))


official.execute_scene_script = observed
official.main()
