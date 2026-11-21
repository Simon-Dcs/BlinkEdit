"""Trusted observer inside private Blender. Record origin at native object construction, before rendering."""

import ast, bpy, json, math, sys, time, traceback
from pathlib import Path
from mathutils import Vector, Matrix, Euler
import bmesh

sys.path.insert(0, '/deps')
SOURCE = '/work/live.py'
records = {}


def scalar(v):
    if type(v) in (int, bool):
        return v
    if type(v) is float and math.isfinite(v):
        return round(v, 5)
    if type(v) is str and len(v) <= 90:
        return v
    if (
        type(v) in (tuple, list)
        and len(v) <= 4
        and all(type(i) in (float, int) and math.isfinite(i) for i in v)
    ):
        return [round(i, 5) for i in v]
    return None


def observe(line, op, result):
    if op.startswith('bpy.ops.'):
        if not isinstance(result, set) or 'FINISHED' not in result:
            return result
        obj = bpy.context.view_layer.objects.active
    else:
        obj = result
    if not isinstance(obj, bpy.types.Object) or obj.type != 'MESH':
        return result
    uid = obj.session_uid
    if uid in records:
        return result
    stack = []
    f = sys._getframe(1)
    while f:
        if f.f_code.co_filename == SOURCE:
            params = {}
            for k in f.f_code.co_varnames[
                : f.f_code.co_argcount + f.f_code.co_kwonlyargcount
            ]:
                v = scalar(f.f_locals.get(k))
                if v is not None:
                    params[k] = v
            stack.append(
                dict(fn=f.f_code.co_name, at=f.f_lineno, params=params)
            )
        f = f.f_back
    records[uid] = dict(native=op, line=line, stack=stack)
    return result


class Instrument(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        op = ast.unparse(node.func)
        if (
            (op.startswith('bpy.ops.mesh.primitive_') and op.endswith('_add'))
            or op
            in (
                'bpy.ops.object.add',
                'bpy.ops.object.duplicate',
                'bpy.ops.object.duplicate_move',
            )
            or op.endswith(('.objects.new', '.copy'))
        ):
            return ast.copy_location(
                ast.Call(
                    ast.Name('__record_mesh_origin__', ast.Load()),
                    [ast.Constant(node.lineno), ast.Constant(op), node],
                    [],
                ),
                node,
            )
        return node


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    tree = Instrument().visit(ast.parse(Path(SOURCE).read_text()))
    ast.fix_missing_locations(tree)
    env = {
        '__name__': '__main__',
        '__file__': SOURCE,
        '__record_mesh_origin__': observe,
        'bpy': bpy,
        'bmesh': bmesh,
        'math': math,
        'Euler': Euler,
        'Matrix': Matrix,
        'Vector': Vector,
    }
    exec(compile(tree, SOURCE, 'exec'), env)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in bpy.context.scene.objects:
        if o.type != 'MESH' or o.hide_render:
            continue
        e = o.evaluated_get(dg)
        v = [e.matrix_world @ Vector(x) for x in e.bound_box]
        out.append(
            dict(
                name=o.name,
                bounds=[
                    [round(min(p[i] for p in v), 5) for i in range(3)],
                    [round(max(p[i] for p in v), 5) for i in range(3)],
                ],
                origin=records.get(o.session_uid),
            )
        )
    Path('/work/objects.json').write_text(
        json.dumps({'objects': out, 'native_events': len(records)})
    )


if __name__ == '__main__':
    main()
