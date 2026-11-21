"""Compose actual fixed-view renders. All cells use the same display size."""

from PIL import Image, ImageDraw, ImageFont
from setup_run import PREV, read
from setup_run import ROOT as R


def font(sz):
    return ImageFont.truetype(
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', sz, index=2
    )


def flat(path):
    i = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', i.size, 'white')
    bg.alpha_composite(i)
    return bg.convert('RGB')


def pick(k, a, kind='edithero'):
    prior = (
        a in ('F0', 'F1') and a not in read(R / 'manifest.json')['run_arms'][k]
    )
    return (
        (PREV if prior else R)
        / kind
        / 'formal'
        / k
        / ('F0R' if prior and a == 'F1' else a)
    )


def main():
    from setup_run import read

    m = read(R / 'manifest.json')
    k = 'R/3ce5c791_s0'
    t = 'turn02'
    im = Image.new('RGB', (1180, 340), '#f5f6f3')
    d = ImageDraw.Draw(im)
    d.text(
        (24, 8),
        '按颜色删除部件：第二轮移除黄色冰棒',
        font=font(25),
        fill='#163c39',
    )
    names = [
        '目标 · 隐藏视角',
        '直接 Codex',
        'F0',
        'F1 · 规则',
        'F2 · 定位后删除',
    ]
    files = [R / 'setup/data/renders' / k / t / 'view2_heldout.png'] + [
        pick(k, a) / t / 'render/f0002.png' for a in m['arms']
    ]
    for j, (name, p) in enumerate(zip(names, files)):
        x = 16 + 232 * j
        d.text((x + 7, 59), name, font=font(18), fill='#385050')
        img = flat(p)
        img.thumbnail((214, 208))
        im.paste(
            img, (x + (222 - img.width) // 2, 92 + (208 - img.height) // 2)
        )
    d.text(
        (22, 304),
        '同一官方相机；各方法接续自己的第一轮结果。',
        font=font(15),
        fill='#536360',
    )
    im.save(R / 'comparisons/popsicle-heldout.jpg', quality=93)
    key = m['fresh']['code'][0]
    im = Image.new('RGB', (1120, 534), '#f5f6f3')
    d = ImageDraw.Draw(im)
    d.text(
        (22, 11),
        '新资产：把最低开放格的顶板向上移动',
        font=font(25),
        fill='#163c39',
    )
    for j, (a, name) in enumerate(
        zip(m['arms'], ['直接 Codex', 'F0', 'F1 · 规则', 'F2 · 局部代码'])
    ):
        x = 74 + 257 * j
        d.text((x, 56), name, font=font(18), fill='#385050')
        q = (
            sorted(pick(key, a, '3dcodeedit').glob('turn*/outcome.json'))[
                -1
            ].parent
            / 'views'
        )
        for i, state in enumerate(['original', 'modified']):
            p = q / (state + '_right.png')
            img = flat(p)
            img.thumbnail((235, 201))
            im.paste(img, (x + (245 - img.width) // 2, 86 + 218 * i))
    for i, t in enumerate(['原始', '最终']):
        d.text((12, 169 + 218 * i), t, font=font(18), fill='#385050')
    d.text(
        (22, 505),
        '各列使用该方法最终评分时的原图配对与共同取景；两行均为能看清隔板的右视角。',
        font=font(15),
        fill='#536360',
    )
    im.save(R / 'comparisons/fresh-shelf.jpg', quality=93)


if __name__ == '__main__':
    main()
