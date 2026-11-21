"""Publication contact sheet: real inputs and outputs, matched views and identical crop."""

from PIL import Image, ImageDraw, ImageFont
from prepare import R, read

m = read(R / 'manifest.json')
fontpath = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def font(size):
    return ImageFont.truetype(fontpath, size, index=2)


def flatten(p):
    a = Image.open(p).convert('RGBA')
    bg = Image.new('RGBA', a.size, 'white')
    bg.alpha_composite(a)
    return bg.convert('RGB')


def draw(rows, out):
    if not rows:
        return
    w, h = 840, 182
    sheet = Image.new('RGB', (w, 64 + h * len(rows)), (248, 249, 247))
    d = ImageDraw.Draw(sheet)
    d.text((20, 16), '实际编辑结果', font=font(23), fill='#20342f')
    for i, t in enumerate(['修改前', '修改后 · F0＋规则']):
        d.text((275 + i * 268, 23), t, font=font(17), fill='#304b41')
    for i, (title, detail, a, b) in enumerate(rows):
        y = 64 + i * h
        d.rectangle((12, y, w - 12, y + h - 6), fill='white')
        d.text((22, y + 49), title, font=font(20), fill='#20342f')
        d.text((22, y + 84), detail, font=font(15), fill='#506c60')
        imgs = list(map(flatten, [a, b]))
        # Same view/crop/scale within each pair, without independently normalizing silhouettes.
        for j, img in enumerate(imgs):
            img.thumbnail((245, 165))
            sheet.paste(
                img,
                (
                    215 + 268 * j + (260 - img.width) // 2,
                    y + (h - img.height) // 2 - 3,
                ),
            )
    sheet.save(out, quality=92)


def main():
    rows = []
    k = 'R/ba34ea4a_s2'
    base = R / 'edithero/formal' / k / 'F0R'
    a = base / 'turn01/render/f0001.png'
    b = base / 'turn02/render/f0001.png'
    if a.exists() and b.exists():
        rows.append(('骑士 · 移除盾牌', '上一轮产物 → 本轮产物', a, b))
    for k, title, detail, view in [
        (
            'openscad_claw_L1_Q0',
            '机械爪 · 独立方块',
            '三条边均放大 1.5 倍',
            'front',
        ),
        (
            '3dcodebench_Fork_seed0_L1_Q0',
            '餐叉 · 缩短叉柄',
            '只修改叉柄长度参数',
            'top',
        ),
        (
            '3dcodebench_Urchin_seed0_L1_Q1',
            '海胆 · 高度压半',
            '先前已知的简单示例',
            'front',
        ),
    ]:
        v = R / '3dcodeedit/formal' / k / 'F0R/turn01/views'
        a = v / f'original_{view}.png'
        b = v / f'modified_{view}.png'
        if a.exists() and b.exists():
            rows.append((title, detail, a, b))
    draw(rows, R / 'comparisons/rule-examples.jpg')


if __name__ == '__main__':
    main()
