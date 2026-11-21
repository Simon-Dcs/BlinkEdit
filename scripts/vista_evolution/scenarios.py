"""Host-only task definitions and independent expected outcomes."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.scene import (
    clone_scene,
    inventory,
)

TEXT = {
    'T1': ['删除办公室里的台灯，餐厅保持不变。'],
    'T2': ['在办公室的桌子上放一个花瓶，其他东西不要移动。'],
    'T3': ['在办公室桌上放台灯、花瓶、书和杯子，不要相互重叠，餐厅保持不变。'],
    'T4': [
        '移除办公室地上的台灯，再在办公室桌上放台灯、花瓶、书和杯子，不要相互重叠，餐厅保持不变。'
    ],
    'C': [
        '在办公室桌上放一个杯子。',
        '在刚才那张桌子上再放一本书，保留杯子。',
        '删掉刚才放的杯子，书不要动。',
        '再删掉刚才那个杯子。',
    ],
    'B1': ['固定方向把超尺寸沙发放到办公室桌上，不要缩放沙发或改变桌子。'],
    'B2': ['删除办公室里的一架钢琴，其他东西不要动。'],
    'B3': ['把花瓶放在那张桌上。'],
    'D1': ['删除办公室地上的台灯，桌上的台灯和餐厅都保持不变。'],
    'D2': [
        '在办公室桌上放一个杯子。',
        '找到刚才放的杯子；如果它已经被挪到餐厅，把它删掉，其他物体保持不变。',
    ],
    'D3': [
        '移除办公室地上的台灯，再在办公室桌上放一个花瓶和一个原尺寸沙发。方向固定；如果不能全部完成，请保持原场景。'
    ],
    'D4': [
        '保留桌上已有的书和杯子，再在办公室桌上放台灯、花瓶、书和杯子，物体不要重叠，餐厅保持不变。'
    ],
}
ORDER = list(TEXT)


def add_link(tree, name, asset, source, xyz):
    root = tree.getroot()
    link = ET.SubElement(root, 'link', name=name)
    for kind, suffix in [('visual', ''), ('collision', '_collision')]:
        part = ET.SubElement(link, kind)
        geo = ET.SubElement(part, 'geometry')
        ET.SubElement(
            geo,
            'mesh',
            filename=str(
                (source / asset / 'mesh' / f'{asset}{suffix}.obj').resolve()
            ),
        )
    joint = ET.SubElement(root, 'joint', name='joint_' + name, type='fixed')
    ET.SubElement(joint, 'parent', link='base')
    ET.SubElement(joint, 'child', link=name)
    ET.SubElement(
        joint, 'origin', xyz=' '.join(map(str, xyz)), rpy='1.5708 0 0'
    )


def prepare(root):
    root = Path(root)
    old = Path('outputs/efficiency/20261112_214800').resolve()
    source = root / 'inputs/source'
    shutil.copytree(old / 'source', source)
    catalog = {
        k: str((source / k / 'mesh' / f'{k}.obj').resolve())
        for k in ['lamp', 'vase', 'book', 'mug', 'sofa']
    }
    write_json(root / 'inputs/catalog.json', catalog)
    cells = []
    offsets = [(0, 0), (0.08, 0.07), (-0.07, -0.10), (0.04, 0.12)]
    for variant, (dx, dy) in enumerate(offsets):
        folder = root / 'inputs' / f'layout_{variant}'
        path = clone_scene(
            old / 'fixtures/two_rooms/scene.urdf', folder / 'base'
        )
        tree = ET.parse(path)
        aliases = {}
        for name in [
            'table_office_001',
            'table_dining_001',
            'lamp_office_001',
            'lamp_dining_001',
        ]:
            if variant < 2:
                aliases[name] = name
            else:
                noun = 'desk' if name.startswith('table') else 'light'
                room = 'office' if 'office' in name else 'dining'
                aliases[name] = (
                    f'{noun}_{room}_{713 + variant * 13 + len(aliases)}'
                )
        for link in tree.findall('link'):
            if link.get('name') in aliases:
                link.set('name', aliases[link.get('name')])
        for joint in tree.findall('joint'):
            child = joint.find('child')
            name = child.get('link')
            if name in aliases:
                child.set('link', aliases[name])
                origin = joint.find('origin')
                p = np.fromstring(origin.get('xyz'), sep=' ')
                p[:2] += [dx, dy]
                origin.set('xyz', ' '.join(map(str, p)))
        tree.write(path, encoding='utf-8', xml_declaration=True)
        objects, _, _ = inventory(path)
        table = objects[aliases['table_office_001']]['mesh'].bounds
        tabletop = float(table[1, 2])
        cx, cy = -1.7 + dx, dy
        d1 = clone_scene(path, folder / 'ambiguous_lamps')
        t = ET.parse(d1)
        floor_id = (
            f'lamp_office_{"a" if variant % 2 == 0 else "b"}_{481 + variant}'
        )
        top_id = (
            f'lamp_office_{"b" if variant % 2 == 0 else "a"}_{481 + variant}'
        )
        for link in t.findall('link'):
            if link.get('name') == aliases['lamp_office_001']:
                link.set('name', floor_id)
        for joint in t.findall('joint'):
            if joint.find('child').get('link') == aliases['lamp_office_001']:
                joint.find('child').set('link', floor_id)
        lamp = trimesh.load(catalog['lamp'], force='mesh')
        lamp.apply_transform(
            trimesh.transformations.euler_matrix(1.5708, 0, 0, 'sxyz')
        )
        add_link(
            t,
            top_id,
            'lamp',
            source,
            [cx, cy, tabletop - float(lamp.bounds[0, 2]) + 0.001],
        )
        t.write(d1, encoding='utf-8', xml_declaration=True)
        d4 = clone_scene(path, folder / 'occupied_table')
        t = ET.parse(d4)
        for asset, x, y in [
            ('book', cx + 0.18, cy + 0.38),
            ('mug', cx - 0.18, cy - 0.36),
        ]:
            mesh = trimesh.load(catalog[asset], force='mesh')
            mesh.apply_transform(
                trimesh.transformations.euler_matrix(1.5708, 0, 0, 'sxyz')
            )
            add_link(
                t,
                f'existing_{asset}_{variant}',
                asset,
                source,
                [x, y, tabletop - float(mesh.bounds[0, 2]) + 0.001],
            )
        t.write(d4, encoding='utf-8', xml_declaration=True)
        for task, instructions in TEXT.items():
            text = instructions.copy()
            if variant >= 2:
                # Equivalent unseen wording, not a different edit request.
                text = [
                    s.replace('删除办公室里的台灯', '把办公室的台灯移除')
                    .replace('餐厅保持不变', '不要改动餐厅')
                    .replace('其他东西不要移动', '其余物体的位置保持原样')
                    .replace('物体不要重叠', '各物体之间留出空隙')
                    for s in text
                ]
            cells.append(
                dict(
                    id=f'{task}_v{variant}',
                    task=task,
                    variant=variant,
                    seed=42 + variant,
                    split='selection' if variant < 2 else 'holdout',
                    instructions=text,
                    input=str(
                        d1 if task == 'D1' else d4 if task == 'D4' else path
                    ),
                    aliases=aliases,
                    floor_lamp=floor_id,
                    table_lamp=top_id,
                )
            )
    write_json(root / 'inputs/cases.json', cells, once=True)
    return cells


def perturb_relocated_cup(source, directory, cell):
    """External editor changes link name and pose, preserving its stable identity."""
    path = clone_scene(source, Path(directory) / 'external_update')
    objects, _, _ = inventory(path)
    cups = [k for k, v in objects.items() if v['asset'] == 'mug']
    if len(cups) != 1:
        return path, dict(
            applied=False, reason='one newly created cup was not present'
        )
    old = cups[0]
    new = f'object_{924 + cell["variant"] * 17}'
    dining = objects[cell['aliases']['table_dining_001']]['mesh'].bounds
    cup = objects[old]['mesh']
    xyz = objects[old]['xyz'].copy()
    xyz[:2] = (dining[0, :2] + dining[1, :2]) / 2
    xyz[2] += dining[1, 2] - cup.bounds[0, 2] + 0.001
    tree = ET.parse(path)
    for link in tree.findall('link'):
        if link.get('name') == old:
            link.set('name', new)
    for joint in tree.findall('joint'):
        if joint.find('child').get('link') == old:
            joint.find('child').set('link', new)
            joint.find('origin').set('xyz', ' '.join(map(str, xyz)))
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return path, dict(
        applied=True,
        old_id=old,
        new_id=new,
        stable_id=objects[old]['link'].get('scene_edit_id'),
    )


def evaluate(cell, turn, before_path, after_path, status, record):
    """All expected IDs stay in this trusted observer, outside model requests."""
    before, _, _ = inventory(before_path)
    after, _, _ = inventory(after_path)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    task = cell['task']
    expected_status = 'execute'
    types = []
    targets = []
    if task == 'T1':
        targets = [cell['aliases']['lamp_office_001']]
    elif task == 'T2':
        types = ['vase']
    elif task == 'T3':
        types = ['lamp', 'vase', 'book', 'mug']
    elif task == 'T4':
        types = ['lamp', 'vase', 'book', 'mug']
        targets = [cell['aliases']['lamp_office_001']]
    elif task == 'C':
        if turn == 0:
            types = ['mug']
        elif turn == 1:
            types = ['book']
        elif turn == 2:
            targets = [k for k, v in before.items() if v['asset'] == 'mug']
        else:
            expected_status = 'noop'
    elif task in ['B1', 'D3']:
        expected_status = 'impossible'
    elif task == 'B2':
        expected_status = 'noop'
    elif task == 'B3':
        expected_status = 'clarify'
    elif task == 'D1':
        targets = [cell['floor_lamp']]
    elif task == 'D2':
        if turn == 0:
            types = ['mug']
        else:
            targets = [k for k, v in before.items() if v['asset'] == 'mug']
    elif task == 'D4':
        types = ['lamp', 'vase', 'book', 'mug']
    changed_ok = sorted(after[k]['asset'] for k in added) == sorted(
        types
    ) and removed == sorted(targets)
    same = all(
        np.array_equal(before[k]['xyz'], after[k]['xyz'])
        and np.array_equal(before[k]['rpy'], after[k]['rpy'])
        and np.array_equal(
            before[k]['mesh'].vertices, after[k]['mesh'].vertices
        )
        for k in set(before) & set(after)
    )
    passed = (
        changed_ok
        and same
        and status == expected_status
        and record.get('common_ok', False)
    )
    critical = bool(set(removed) - set(targets)) or not same
    return dict(
        passed=bool(passed),
        critical=bool(critical),
        expected_status=expected_status,
        status=status,
        added=added,
        removed=removed,
        expected_assets=types,
        expected_removed=targets,
        preserved=same,
    )
