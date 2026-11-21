"""Conservative rule router: sees the current asset, user instruction and own action history only."""

import ast
import re


def words(text):
    return set(re.findall('[a-z]+', (text or '').lower()))


def singular(w):
    return w[:-1] if w.endswith('s') and not w.endswith('ss') else w


def tags(text):
    return {singular(w) for w in words(text)} - {
        'slot',
        'anchor',
        'part',
        'mesh',
        'the',
        'a',
        'of',
        'and',
        'on',
        'with',
        'from',
        'to',
        'its',
        'in',
        'at',
        'for',
        'by',
    }


def removed_doc(g, ops, before, prior):
    """Serialize just confirmed operations; archived primitive IDs reference the same GLB binary."""
    h = dict(prior)
    for op in ops:
        if op['op'] == 'remove':
            k = str(int(op['id']))
            node = g['nodes'][int(k)]
            h[k] = dict(
                name=before['nodes'][k]['name'],
                mesh=node['mesh'],
                matrix=before['nodes'][k]['world'],
                parent=before['nodes'][k]['parent'],
            )
        elif op['op'] == 'restore':
            h.pop(str(op['id']), None)
    return h


def additions_doc(text, ops, before, after, prior):
    # Remember groups actually created, not a textual list of wishes.
    ids = sorted(set(after['nodes']) - set(before['nodes']), key=int)
    keep = [a for a in prior if all(k in after['nodes'] for k in a['ids'])]
    if (
        ids
        and all(o['op'] == 'add' for o in ops)
        and re.match(
            r'^(?:add|place|set|stand|attach) (?:a |an |the )', text.lower()
        )
    ):
        keep.append(
            dict(
                text=text.lower(),
                ids=ids,
                names={k: after['nodes'][k].get('name') for k in ids},
            )
        )
    return keep


def eh(text, state, source, archived, created=()):
    from embodied_gen.harness.asset_edit import gltf

    q = text.lower().strip().rstrip('.')
    ns = state['nodes']
    m = re.fullmatch(
        r'reinstall the ([a-z]+) (?:in its mounting position|in its original position|back in place)',
        q,
    )
    if m:
        matches = [
            (k, a) for k, a in archived.items() if m[1] in tags(a['name'])
        ]
        if len(matches) != 1:
            return None
        k, a = matches[0]
        g = gltf.header(source)[0]
        # Restrict restoration to unchanged root nodes and still-retained meshes.
        roots = g['scenes'][g.get('scene', 0)]['nodes']
        if (
            a['parent'] is not None
            or int(k) not in roots
            or 'mesh' in g['nodes'][int(k)]
            or g['nodes'][int(k)].get('name') != a['name']
            or not 0 <= a['mesh'] < len(g['meshes'])
        ):
            return None
        return [dict(op='restore', id=int(k), mesh=a['mesh'])]
    m = re.fullmatch(
        r'remove the (?:[a-z]+[’\']s )?([a-z]+(?: [a-z]+){0,4})', q
    )
    if len(ns) < 2:
        return None
    # Bind a reference to the saved parts of an actual earlier addition.
    q_simple = q.removeprefix('remove the ')
    if q.startswith('remove the ') and not (
        words(q_simple) & {'left', 'right', 'all', 'both', 'remaining'}
    ):
        desc = re.split(r' from (?:the |its )', q_simple, maxsplit=1)[0]
        phrase = re.escape(desc) + r'(?:[ .,;]|$)'
        groups = [
            a
            for a in created
            if re.match(
                r'^(?:add|place|set|stand|attach) (?:a |an |the )' + phrase,
                a['text'],
            )
        ]
        if len(groups) == 1 and len(groups[0]['ids']) < len(ns):
            a = groups[0]
            # Asset-native labels must also identify the subject; no other
            # active part may have the same subject. Include every created mesh.
            subj = desc.split(' with ')[0].split()[-1]
            others = set(ns) - set(a['ids'])
            if (
                any(subj in tags(n) for n in a['names'].values())
                and not any(subj in tags(ns[k].get('name')) for k in others)
                and all(
                    k in ns and ns[k].get('name') == a['names'][k]
                    for k in a['ids']
                )
            ):
                return [dict(op='remove', id=int(k)) for k in a['ids']]
    if not m:
        return None
    ts = tags(m[1])
    # Never infer sidedness from label alone (viewer's left may not be asset left).
    if ts & {
        'left',
        'right',
        'remaining',
        'top',
        'bottom',
        'all',
        'both',
        'tallest',
        'highest',
    }:
        return None
    match = [k for k, n in ns.items() if ts and ts <= tags(n.get('name'))]
    if len(match) != 1:
        return None
    k = match[0]
    g = gltf.header(source)[0]
    if g['nodes'][int(k)].get('children'):
        return None
    return [dict(op='remove', id=int(k))]


def _literal_parameter(source, noun, dimension, factor):
    tree = ast.parse(source)
    names = {noun + '_' + dimension, dimension + '_' + noun}
    matches = [
        x
        for x in ast.walk(tree)
        if isinstance(x, ast.Assign)
        and len(x.targets) == 1
        and isinstance(x.targets[0], ast.Name)
        and x.targets[0].id.lower() in names
    ]
    if len(matches) != 1:
        return None
    a = matches[0]
    v = a.value
    if (
        not isinstance(v, ast.Constant)
        or type(v.value) not in (int, float)
        or v.value <= 0
        or v.lineno != v.end_lineno
    ):
        return None
    if not any(
        isinstance(n, ast.Name)
        and isinstance(n.ctx, ast.Load)
        and n.id == a.targets[0].id
        for n in ast.walk(tree)
    ):
        return None
    lines = source.splitlines(keepends=True)
    line = lines[v.lineno - 1]
    lines[v.lineno - 1] = (
        line[: v.col_offset] + str(v.value * factor) + line[v.end_col_offset :]
    )
    return ''.join(lines)


def ce(text, source):
    """Single-operation grammar; no task ID, checklist or reference source diff."""
    clauses = text.lower().strip().rstrip('.').split('; ')
    if (
        len(clauses) != 2
        or clauses[0]
        != "preserve the object's identity and parts outside the requested edits"
    ):
        return None
    q = clauses[1]
    whole = re.fullmatch(
        r'compress the whole ([a-z]+) to (?:about )?half its original (?:pole-to-pole )?height while retaining its original equatorial diameter',
        q,
    )
    cube = re.fullmatch(
        r'make the detached cubic block uniformly (one-and-a-half|two) times larger in every dimension',
        q,
    )
    scalar = re.fullmatch(
        r'(?:make|shorten|reduce) (?:the |all (?:four |three |two )?)([a-z]+) (?:conspicuously shorter, reducing it to |(?:to )?)(?:(?:about|approximately|roughly) )?half (?:its|their) original (length|height)',
        q,
    )
    if scalar:
        program = _literal_parameter(
            source, singular(scalar[1]), scalar[2], 0.5
        )
        if program:
            return program, 'scalar'
    if not (whole or cube):
        return None
    selector = 'whole' if whole else 'cube'
    factor = (
        (1, 1, 0.5)
        if whole
        else ((1.5,) * 3 if cube[1] == 'one-and-a-half' else (2.0,) * 3)
    )
    name = whole[1] if whole else 'cube'
    tail = """
import bpy,re
from mathutils import Matrix,Vector
bpy.context.view_layer.update()
_parts=[o for o in bpy.context.scene.objects if o.type=='MESH' and not o.hide_render]
_spec=SPEC
if _spec[0]=='whole':
    _hits=[o for o in _parts if len(_parts)==1 and o.name.lower().removesuffix('factory')==_spec[1]]
else:
    _hits=[o for o in _parts if 'cube' in re.findall('[a-z]+',o.name.lower()) and max(o.dimensions)>0 and min(o.dimensions)/max(o.dimensions)>.99]
if len(_hits)!=1 or _hits[0].children:raise ValueError('Unresolved unique target')
_o=_hits[0]; _box=[_o.matrix_world@Vector(p) for p in _o.bound_box]
_c=sum(_box,Vector())/8
_o.matrix_world=Matrix.Translation(_c)@Matrix.Diagonal((*_spec[2],1))@Matrix.Translation(-_c)@_o.matrix_world
bpy.context.view_layer.update()
"""
    return source + '\n' + tail.replace(
        'SPEC', repr((selector, name, factor))
    ), 'object'
