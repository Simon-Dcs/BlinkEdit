"""Select patch locations from meshes that survive an actual execution of the current program."""

import ast, collections, re
import numpy as np
from . import local_program

# Descriptive aliases, not benchmark IDs. A matched name must also have a live construction site.
ALIAS = {
    'partition': 'divider',
    'dividers': 'divider',
    'partitions': 'divider',
    'shelves': 'shelf',
    'shelving': 'shelf',
    'legs': 'leg',
    'handles': 'handle',
    'grip': 'handle',
    'rungs': 'rung',
    'doors': 'door',
    'feet': 'foot',
    'teeth': 'tooth',
    'panels': 'panel',
}
IGNORE = set(
    'the an a in of with to and while from for original their its this at that objects preserve object identity parts outside requested edits make reduce raise lower rotate scale by all each conspicuously noticeably visibly markedly approximately roughly half four five three two one remaining interior which it so than much others as length height width thick long narrow tall wide bottom right front middle rear entire whole single longer shorter horizontal vertical upwards upward downward between before after size degrees following unchanged'.split()
)
NUM = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8}
ANC = {'liner', 'lining', 'skin', 'trim', 'face'}


def words(s):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s).lower()
    ts = re.findall('[a-z]+', s)
    out = {
        ALIAS.get(t, t[:-1] if t.endswith('s') and len(t) > 4 else t)
        for t in ts
    }
    if 'bay' in out and 'wall' in out:
        out.add('divider')
    return out


def dims(obj):
    a, b = np.array(obj['bounds'])
    return b - a, (b + a) / 2


def _site(s):
    # Secondary callers identify the part; the leaf is normally a shared primitive helper.
    stack = s['origin']['stack']
    return stack[1]['at'] if len(stack) > 1 else stack[0]['at']


def _family(o):
    return re.sub(r'\d+', '#', o['name'].lower())


def select(src, text, feedback, trace):
    clauses = [
        x.strip()
        for x in text.split(';')
        if 'preserve the object' not in x.lower()
    ]
    if len(clauses) != 1:
        return None, {'reason': 'multiple_targets'}
    q = clauses[0]
    terms = words(q) - IGNORE
    objs = trace['objects']
    groups = collections.defaultdict(list)
    for obj in objs:
        groups[_family(obj)].append(obj)
    count = next(
        (v for n, v in NUM.items() if re.search(r'\b' + n + r'\b', q.lower())),
        None,
    )
    pool = []
    for family, os in groups.items():
        tags = words(family)
        matches = tags & terms
        if not matches or (count and len(os) != count):
            continue
        if tags & ANC and not terms & ANC:
            continue
        if not all(o['origin'] and o['origin']['stack'] for o in os):
            continue
        # A broad common name alone (e.g. 'panel' in many families) cannot win a tie.
        specificity = sum(
            1.0 / sum(w in words(f) for f in groups) for w in matches
        )
        pool.append((len(matches) + specificity, family, os))
    pool.sort(key=lambda x: (-x[0], x[1]))
    if not pool or (len(pool) > 1 and pool[0][0] - pool[1][0] < 0.20):
        return None, {
            'reason': 'ambiguous_or_unmapped',
            'candidate_families': [v[1] for v in pool[:6]],
        }
    _, family, targets = pool[0]
    if (
        not count
        and len(targets) > 1
        and not re.search(r'\b(all|each|every|both)\b', q, re.I)
    ):
        return None, {'reason': 'ambiguous_instances', 'family': family}
    target_names = {o['name'] for o in targets}
    extra = []
    # Include thin attached face liners only when aligned to the same target in space.
    for o in objs:
        if (
            o['name'] in target_names
            or not (words(o['name']) & ANC)
            or not o['origin']
        ):
            continue
        size, ctr = dims(o)
        for t in targets:
            ts, tc = dims(t)
            ax = int(np.argmin(ts))
            other = [j for j in range(3) if j != ax]
            if (
                np.all(
                    np.abs(ctr[other] - tc[other])
                    <= 0.12 * np.maximum(ts[other], 1e-5)
                )
                and np.all(size[other] >= 0.75 * ts[other])
                and abs(ctr[ax] - tc[ax]) <= 1.3 * max(ts[ax], 1e-5)
                and size[ax] <= ts[ax]
            ):
                extra.append(o)
                break
    hits = targets + extra
    sites = sorted({_site(o) for o in hits})
    tree = ast.parse(src)
    lines = src.splitlines()
    # Include whole call statement, enclosing loop header and nearby argument definitions.
    ranges = []
    for ln in sites:
        containing = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.stmt) and n.lineno <= ln <= n.end_lineno
        ]
        if not containing:
            return None, {'reason': 'unlocated_statement'}
        small = min(containing, key=lambda n: n.end_lineno - n.lineno)
        scope = min(
            (
                n
                for n in containing
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            key=lambda n: n.end_lineno - n.lineno,
            default=None,
        )
        lo = max(scope.lineno + 1 if scope else 1, small.lineno - 8)
        hi = min(
            scope.end_lineno if scope else len(lines), small.end_lineno + 5
        )
        ranges.append((lo, hi))
    spans = []
    for a, b in sorted(ranges):
        if spans and a <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))
    if sum(len(x) + 8 for a, b in spans for x in lines[a - 1 : b]) > 6000:
        return None, {'reason': 'wide_edit'}
    proof = [
        {
            'name': o['name'],
            'site': _site(o),
            'builder': o['origin']['stack'][0]['fn'],
            'arguments': o['origin']['stack'][0]['params'],
            'bounds': o['bounds'],
        }
        for o in hits
    ]
    return spans, {
        'reason': 'live_object_family',
        'target_names': sorted(target_names),
        'related': [x['name'] for x in extra],
        'sites': sites,
        'proof': proof,
    }
