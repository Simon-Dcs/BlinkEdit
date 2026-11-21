"""Resolve a singular removal against the *current* embedded GLB. No reference assets."""

import colorsys
import io
import re

import numpy as np
from PIL import Image, ImageDraw

from . import gltf

# Common hue directions in sRGB degrees; gray/white/black require different evidence.
HUES = {
    'red': 0,
    'orange': 30,
    'yellow': 60,
    'green': 120,
    'cyan': 180,
    'blue': 235,
    'purple': 280,
    'pink': 335,
}
DT = {
    5120: 'i1',
    5121: 'u1',
    5122: '<i2',
    5123: '<u2',
    5125: '<u4',
    5126: '<f4',
}
DIM = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def accessor(g, b, k):
    a = g['accessors'][k]
    if 'sparse' in a or 'bufferView' not in a:
        raise ValueError('sparse accessor')
    v = g['bufferViews'][a['bufferView']]
    dt = np.dtype(DT[a['componentType']])
    cols = DIM[a['type']]
    values = np.ndarray(
        (a['count'], cols),
        dtype=dt,
        buffer=b,
        offset=v.get('byteOffset', 0) + a.get('byteOffset', 0),
        strides=(v.get('byteStride', cols * dt.itemsize), dt.itemsize),
    )
    if a.get('normalized'):
        return values.astype(float) / np.iinfo(dt).max
    return values


def _linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(
        x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1 / 2.4)) - 0.055
    )


def _singular(word):
    return word[:-1] if word.endswith('s') and len(word) > 3 else word


def tokens(t):
    return {_singular(w) for w in re.findall('[a-z]+', (t or '').lower())}


def candidates(text, state, g):
    m = re.fullmatch(
        r'(?:remove|delete|detach|take off) (?:the |a |an )(.+?)[.]?',
        text.strip().lower(),
    )
    if not m or re.search(
        r'\b(and|both|all|two|three|pair|set of|except)\b', m[1]
    ):
        return []
    # Narrow by the object noun; color, ordinal and reference words are not identities.
    desc = re.split(r'\b(?:from|with|which|that|on)\b', m[1])[0]
    ws = (
        tokens(desc)
        - set(HUES)
        - {
            'the',
            'a',
            'an',
            'of',
            'its',
            'left',
            'right',
            'remaining',
            'highest',
            'lowest',
            'tallest',
            'widest',
            'small',
            'large',
            'dark',
            'light',
            'brown',
            'black',
            'white',
            'gray',
        }
    )
    ns = state['nodes']
    eligible = [k for k in ns if not g['nodes'][int(k)].get('children')]
    scores = {k: len(ws & tokens(ns[k]['name'])) for k in eligible}
    top = max(scores.values(), default=0)
    ids = [k for k in eligible if top > 0 and scores[k] == top]
    return ids if 1 <= len(ids) <= 12 and len(ns) > 1 else []


def triangles(g, blob, state, key, cache):
    """Sample actual UV islands at triangle centroids; retain world-area weights."""
    n = g['nodes'][int(key)]
    M = np.asarray(state['nodes'][key]['world'])
    geoms = []
    cols = []
    complete = True
    for p in g['meshes'][n['mesh']]['primitives']:
        if p.get('mode', 4) != 4:
            raise ValueError('not triangle geometry')
        attr = p['attributes']
        v = accessor(g, blob, attr['POSITION'])
        idx = (
            accessor(g, blob, p['indices']).ravel()
            if 'indices' in p
            else np.arange(len(v))
        ).reshape(-1, 3)
        # Fixed, bounded sample for large imported meshes.
        idx = idx[:: max(1, (len(idx) + 3999) // 4000)]
        tri = v[idx] @ M[:3, :3].T + M[:3, 3]
        mat = g.get('materials', [])[p['material']] if 'material' in p else {}
        pbr = mat.get('pbrMetallicRoughness', {})
        c = np.tile(
            np.asarray(pbr.get('baseColorFactor', [1] * 4))[:3], (len(idx), 1)
        ).astype(float)
        tex = pbr.get('baseColorTexture')
        if tex:
            tx = g['textures'][tex['index']]
            image = tx['source']
            uvkey = f"TEXCOORD_{tex.get('texCoord', 0)}"
            if uvkey not in attr or tex.get('extensions'):
                complete = False
            else:
                if image not in cache:
                    bv = g['bufferViews'][g['images'][image]['bufferView']]
                    off = bv.get('byteOffset', 0)
                    cache[image] = np.asarray(
                        Image.open(
                            io.BytesIO(blob[off : off + bv['byteLength']])
                        ).convert('RGB')
                    )
                im = cache[image]
                h, w = im.shape[:2]
                uv = accessor(g, blob, attr[uvkey])[idx].mean(axis=1)
                samp = (
                    g.get('samplers', [])[tx['sampler']]
                    if 'sampler' in tx
                    else {}
                )
                for j, wrap in enumerate(['wrapS', 'wrapT']):
                    mode = samp.get(wrap, 10497)
                    uv[:, j] = (
                        np.clip(uv[:, j], 0, 1)
                        if mode == 33071
                        else 1 - np.abs(uv[:, j] % 2 - 1)
                        if mode == 33648
                        else uv[:, j] % 1
                    )
                c *= _linear(
                    im[
                        np.minimum((uv[:, 1] * h).astype(int), h - 1),
                        np.minimum((uv[:, 0] * w).astype(int), w - 1),
                    ]
                    / 255.0
                )
        if 'COLOR_0' in attr:
            c *= accessor(g, blob, attr['COLOR_0'])[idx][..., :3].mean(axis=1)
        geoms.append(tri)
        cols.append(_srgb(c))
    return np.concatenate(geoms), np.concatenate(cols), complete


def hue_support(tri, c, hue):
    hsv = np.array([colorsys.rgb_to_hsv(*row) for row in c])
    distance = np.abs((hsv[:, 0] * 360 - hue + 180) % 360 - 180)
    area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    if not np.any(area):
        return 0.0
    return float(
        np.average(
            (distance <= 17) & (hsv[:, 1] >= 0.42) & (hsv[:, 2] >= 0.32),
            weights=area,
        )
    )


def _panel(samples, path):
    """Current-state views; common orientation and relative size within each view."""
    n = len(samples)
    im = Image.new('RGB', (640, 228 + 104 * ((n + 3) // 4)), (244, 246, 247))
    draw = ImageDraw.Draw(im)
    axes = np.array(
        [[0.82, 0, -0.57], [-0.20, -0.93, -0.29], [0.53, -0.36, 0.76]]
    ).T

    def plot(keys, box, labels):
        pts = (
            np.concatenate([samples[k][0].reshape(-1, 3) for k in keys]) @ axes
        )
        low, hi = pts[:, :2].min(0), pts[:, :2].max(0)
        x, y, w, h = box
        scale = min(
            w / max(hi[0] - low[0], 1e-8), h / max(hi[1] - low[1], 1e-8)
        )
        offset = np.array([x + w / 2, y + h / 2]) - (hi + low) * scale / 2
        faces = []
        centers = {}
        for k in keys:
            t, c, _ = samples[k]
            t = t @ axes
            xy = t[:, :, :2] * scale + offset
            centers[k] = xy.reshape(-1, 2).mean(0)
            faces.extend(
                zip(
                    t[:, :, 2].mean(1),
                    xy.astype(int).reshape(-1, 6).tolist(),
                    (c * 255).astype(int).tolist(),
                )
            )
        for _, coords, col in sorted(faces, key=lambda v: v[0]):
            draw.polygon(coords, fill=tuple(col))
        if labels:
            for k, (u, v) in centers.items():
                draw.rectangle(
                    (u - 4, v - 2, u + len(k) * 7 + 2, v + 10), fill='white'
                )
                draw.text((u - 1, v - 1), k, fill='black')

    draw.text(
        (12, 6),
        'Current candidates (+Y up), same world frame. Numbers are node IDs.',
        fill='black',
    )
    plot(list(samples), (20, 25, 600, 181), True)
    for j, k in enumerate(samples):
        x = j % 4 * 160
        y = 220 + j // 4 * 104
        draw.text((x + 4, y), 'ID ' + k, fill='black')
        plot([k], (x + 18, y + 15, 128, 78), False)
    im.save(path)


def propose(text, state, source, out):
    g, pos = gltf.header(source)
    keys = candidates(text, state, g)
    if not keys:
        return None
    with source.open('rb') as f:
        f.seek(pos)
        blob = f.read()
    try:
        cache = {}
        samples = {k: triangles(g, blob, state, k, cache) for k in keys}
    except (KeyError, ValueError, IndexError, NotImplementedError):
        return None
    colors = tokens(text) & HUES.keys()
    evidence = {'ids': keys}
    # Exact short color+noun removal only. Extra constraints need a separate decision.
    if (
        len(colors) == 1
        and re.fullmatch(
            r'(?:remove|delete|detach) the [a-z]+ [a-z]+[.]?', text.lower()
        )
        and all(v[2] for v in samples.values())
    ):
        color = next(iter(colors))
        scores = {
            k: hue_support(t, c, HUES[color])
            for k, (t, c, _) in samples.items()
        }
        evidence['color_surface_fraction'] = scores
        order = sorted(keys, key=scores.get, reverse=True)
        best = order[0]
        runner = scores[order[1]] if len(order) > 1 else 0.0
        if scores[best] >= 0.45 and scores[best] - runner >= 0.25:
            return 'measured', int(best), evidence
    panel = out / 'current-candidates.png'
    _panel(samples, panel)
    return 'ask', panel, evidence


def selected(reply, ids):
    import json

    try:
        val = json.loads(
            reply.strip().removeprefix('```json').removesuffix('```').strip()
        )['id']
    except (KeyError, TypeError, ValueError):
        return None
    return val if type(val) is int and str(val) in ids else None


def verify_removed(before, after, k):
    expected = set(before['nodes']) - {str(k)}

    def native(n):
        return {k: v for k, v in n.items() if k != 'stable_id'}

    if set(after['nodes']) != expected or any(
        native(before['nodes'][i]) != native(after['nodes'][i])
        for i in expected
    ):
        raise ValueError('selected removal changed a non-target part')
