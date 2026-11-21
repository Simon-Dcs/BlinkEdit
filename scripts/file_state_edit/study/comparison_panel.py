"""Held-out EditHero frame 1, unaltered conditioning; fixed representative slots."""

from pathlib import Path
from PIL import Image, ImageDraw
from prepare import R
from panels import font, flatten


def main():
    rows = [
        ('骑士 · 移除盾牌', '简单组，第 2 轮', 'R/ba34ea4a_s2', 2),
        ('船 · 移除棕色皮箱', '简单组，第 5 轮', 'G/f6ff3815_s1', 5),
        ('船 · 原位复装船舱', '简单组，第 10 轮', 'X/962ef577_s0', 10),
        ('更换鞋靴', '一般组 P，第 1 轮', 'P/933e8e46_s1', 1),
        ('树冠 · 局部改材质', '一般组 M，第 1 轮', 'M/914b7fc7_s0', 1),
    ]
    w = 980
    h = 165
    canvas = Image.new('RGB', (w, 62 + len(rows) * h), (248, 249, 247))
    d = ImageDraw.Draw(canvas)
    for j, l in enumerate(['参考 · 隐藏视角', 'F0', 'F0＋规则', 'F1＋规则']):
        d.text((240 + j * 180, 17), l, font=font(17), fill='#23483d')
    for i, (name, detail, key, t) in enumerate(rows):
        y = 62 + i * h
        d.rectangle((10, y, w - 10, y + h - 5), fill='white')
        d.text((18, y + 44), name, font=font(18), fill='#20342f')
        d.text((18, y + 78), detail, font=font(14), fill='#506c60')
        paths = [
            R / 'setup/data/renders' / key / f'turn{t:02}/view1_heldout.png'
        ] + [
            R / 'edithero/formal' / key / a / f'turn{t:02}/render/f0001.png'
            for a in ['F0', 'F0R', 'F1R']
        ]
        for j, p in enumerate(paths):
            if p.exists():
                im = flatten(p)
                im.thumbnail((165, 153))
                canvas.paste(
                    im, (218 + j * 180 + (180 - im.width) // 2, y + 4)
                )
            else:
                d.text(
                    (245 + j * 180, y + 66),
                    '无该轮产物',
                    font=font(14),
                    fill='#84918a',
                )
    canvas.save(R / 'comparisons/heldout-methods.jpg', quality=92)


if __name__ == '__main__':
    main()
