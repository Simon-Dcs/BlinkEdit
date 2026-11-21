"""Find source spans related to a requested part and apply checked line edits to those spans."""

import ast
import json
import math
import re
from collections import Counter

STOP = set(
    'the its this that from into with and for not all only each original modified requested instruction preserve object identity parts outside edits should rather than while visibly visible still source scene mesh python blender make must has are geometry create add bpy data objects self return name original modified pass fail front view views model same part and shape current other new seed no side renderer'.split()
)


def words(s):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return {
        w[:-1] if w.endswith('s') and len(w) > 3 else w
        for w in re.findall('[a-z]+', s.lower())
        if len(w) > 2 and w not in STOP
    }


def windows(src, instruction, feedback=''):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    lines = src.splitlines()
    blocks = []

    def add(node, fullname):
        start = node.lineno
        if getattr(node, 'decorator_list', []):
            start = min(start, min(x.lineno for x in node.decorator_list))
        blocks.append((start, node.end_lineno, fullname))

    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            for c in n.body:
                if isinstance(c, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
                    add(
                        c,
                        n.name
                        + '.'
                        + getattr(
                            c,
                            'name',
                            lines[c.lineno - 1].strip().split('=')[0][:75],
                        ),
                    )
        elif isinstance(n, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            add(
                n,
                getattr(
                    n, 'name', lines[n.lineno - 1].strip().split('=')[0][:75]
                ),
            )
    if not blocks:
        return None
    q = words(instruction)
    fb = words(feedback)
    names = [words(name) for _, _, name in blocks]
    bodies = [words('\n'.join(lines[a - 1 : b])) for a, b, _ in blocks]
    freq = Counter(w for bag in bodies for w in bag)

    def val(bag):
        return sum(
            (1.0 if w in q else 0.35) * math.log1p(len(blocks) / (freq[w] + 1))
            for w in bag & (q | fb)
        )

    ranking = sorted(
        range(len(blocks)),
        key=lambda i: (
            -(
                2.5 * val(names[i])
                + val(bodies[i]) / math.sqrt(max(1, len(bodies[i]) / 22))
            )
        ),
    )
    chosen = []
    size = 0
    for i in ranking:
        a, b, name = blocks[i]
        matches = (q | fb) & bodies[i]
        # Generic entry points, imports and ubiquitous geometry helpers are poor patch targets.
        if not matches or not any(
            freq[w] <= max(2, len(blocks) * 0.6) for w in matches
        ):
            continue
        if b - a > 85:
            hit = sorted(
                range(a, b + 1), key=lambda j: -val(words(lines[j - 1]))
            )[0]
            a, b = max(a, hit - 18), min(b, hit + 21)
        length = sum(len(x) + 8 for x in lines[a - 1 : b])
        if size + length > 6000:
            continue
        chosen.append((a, b))
        size += length
        if len(chosen) >= 6:
            break
    if not chosen or len(src) < 550 or size > len(src) * 0.85:
        return None
    return sorted(chosen)


def patch_prompt(src, spans, instruction, feedback):
    lines = src.splitlines()
    return (
        'Edit only the relevant lines of this existing Blender Python program. The host assembles and renders the complete program. '
        'Do not redescribe unchanged code. Numbers on the left are original 1-based inclusive line numbers, not code. '
        'Return only JSON {"edits":[{"start":17,"end":18,"code":"two replacement lines with exact indentation"}]}. '
        'Use distinct, nonoverlapping ranges inside the excerpts. You may include new lines inside a replacement. '
        'If required definitions or changes lie outside these windows, return {"edits":[]} so the full editor can take over. '
        'Implement every requested clause and preserve other parts.\nInstruction: '
        + instruction
        + (
            '\nFeedback about the current program: ' + feedback
            if feedback
            else ''
        )
        + '\nCurrent program excerpts:\n'
        + '\n...\n'.join(
            '\n'.join(f'{j}: {lines[j - 1]}' for j in range(a, b + 1))
            for a, b in spans
        )
    )


def apply_reply(src, spans, reply):
    try:
        edits = json.loads(
            reply.strip().removeprefix('```json').removesuffix('```').strip()
        )['edits']
        if not isinstance(edits, list) or not 1 <= len(edits) <= 20:
            return None
        changes = []
        for e in edits:
            a, b, c = e['start'], e['end'], e['code']
            if (
                type(a) is not int
                or type(b) is not int
                or not isinstance(c, str)
                or not any(lo <= a <= b <= hi for lo, hi in spans)
            ):
                return None
            changes.append((a, b, c))
        changes.sort()
        if any(r[0] <= l[1] for l, r in zip(changes, changes[1:])):
            return None
        lines = src.splitlines(keepends=True)
        for a, b, c in reversed(changes):
            lines[a - 1 : b] = (
                [c + ('' if c.endswith('\n') else '\n')] if c else []
            )
        new = ''.join(lines)
        ast.parse(new)
        return new if new != src else None
    except (KeyError, IndexError, TypeError, ValueError, SyntaxError):
        return None
