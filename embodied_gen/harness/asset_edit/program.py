"""Version-specific index; Blender builders can recreate all geometry on every run."""

import ast


def observe(code):
    tree = ast.parse(code)
    functions = {
        n.name: sorted(
            {
                v.id
                for v in ast.walk(n)
                if isinstance(v, ast.Name) and isinstance(v.ctx, ast.Load)
            }
        )
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    parameters = {}
    for n in tree.body:
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ):
            try:
                val = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                continue
            if isinstance(val, (int, float, str, bool)):
                parameters[n.targets[0].id] = val
    return dict(
        schema='blender-program-v1',
        axes='Blender local/world, Z up',
        parameters=parameters,
        functions={
            k: sorted(set(v) & (functions.keys() | parameters.keys()))
            for k, v in functions.items()
        },
        generated_part_mapping=None,
    )
