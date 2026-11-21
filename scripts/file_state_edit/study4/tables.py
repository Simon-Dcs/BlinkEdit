"""Render the report's final tables only after every frozen output has been graded."""

from design import R
from metrics import collect


def fmt(v):
    return f'{v:,}'


def render():
    groups, cases, turns, trials, pairs, budget = collect(final=True)
    dest = R / 'report/program-tables.md'
    rows = [
        '| 方法 | 首稿接受 | 修正次数 | 最终全通过 | 编辑 tokens | 反馈 tokens | 交付 tokens | 交付秒数 | 盲评 tokens |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for a, c in groups['all'].items():
        rows.append(
            f"| {a} | {c['first_accept']}/24 | {c['repairs']} | {c['passed']}/24 | {fmt(c['editor_tokens'])} | {fmt(c['reviewer_tokens'])} | {fmt(c['delivery'])} | {c['service_s']:,.1f} | {fmt(c['verifier_tokens'])} |"
        )
    rows += [
        '',
        '每格依次为**最终全通过题数；交付 tokens**。',
        '',
        '| 样本 | F1 | F2 | F2T |',
        '|---|---:|---:|---:|',
    ]
    for key, name in [
        ('simple', '原简单候选（8 题）'),
        ('general', '原一般题（12 题）'),
        ('main', '原 20 题合计'),
        ('fresh', '新源程序（4 题）'),
    ]:
        vals = [
            f"{c['passed']}/{c['assigned']}；{fmt(c['delivery'])}"
            for c in groups[key].values()
        ]
        rows.append('| ' + ' | '.join([name, *vals]) + ' |')
    dest.write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    render()
