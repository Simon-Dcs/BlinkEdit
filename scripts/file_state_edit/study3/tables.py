"""Write report tables from final ledgers, including failures and matched complete chains."""

from evidence import analyse
from setup_run import ROOT

M = ['direct', 'F0', 'F1', 'F2']
LABEL = {'direct': '直接 Codex', 'F0': 'F0', 'F1': 'F1', 'F2': 'F2'}


def num(v, d=0):
    return (
        '—'
        if v is None
        else (f'{v:,}' if isinstance(v, int) and d == 0 else f'{v:,.{d}f}')
    )


def table(columns, rows):
    return '\n'.join(
        [
            '| ' + ' | '.join(columns) + ' |',
            '|' + '|'.join(['---'] + ['---:'] * (len(columns) - 1)) + '|',
            *['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows],
        ]
    )


def build():
    t, c, q, g, p, e = analyse()
    out = ROOT / 'report'
    out.mkdir(exist_ok=True)
    labels = {'simple': '简单候选', 'general': '一般', 'fresh': '新资产'}
    rows = []
    for pop, n in [('simple', 75), ('general', 18), ('fresh', 9)]:
        for a in M:
            z = g['edithero/' + pop][a]
            rows.append(
                [
                    labels[pop] + ' · ' + LABEL[a],
                    f"{z['complete']}/{z['assigned']}",
                    f"{z['done']}/{n}",
                    num(z['IF'], 3),
                    num(z['CC'], 3),
                    num(z['MAE'], 4),
                    z['editor_calls'],
                    num(z['editor_tokens']),
                    num(z['service_s'], 1),
                ]
            )
    (out / 'eh-table.md').write_text(
        table(
            [
                '样本与方法',
                '完成链',
                '交付轮',
                '指令完成 IF ↑',
                '保持 CC ↑',
                '颜色误差 ↓',
                '编辑请求',
                '编辑 tokens',
                '编辑秒数',
            ],
            rows,
        )
    )
    rows = []
    for pop in labels:
        for a in M:
            z = g['3dcodeedit/' + pop][a]
            rows.append(
                [
                    labels[pop] + ' · ' + LABEL[a],
                    f"{z['first']}/{z['assigned']}",
                    z['repairs'],
                    f"{z['passed']}/{z['judged']}",
                    num(z['editor_tokens']),
                    num(z['reviewer_tokens']),
                    num(z['editor_tokens'] + z['reviewer_tokens']),
                    num(z['service_s'], 1),
                    num(z['verifier_tokens']),
                    num(z['verifier_latency'], 1),
                ]
            )
    (out / 'ce-table.md').write_text(
        table(
            [
                '样本与方法',
                '首稿接受',
                '修正次数',
                '全通过 / 已盲评',
                '编辑 tokens',
                '反馈 tokens',
                '交付 tokens',
                '交付秒数',
                '独立评分 tokens',
                '评分模型秒数',
            ],
            rows,
        )
    )
    return t, c, q, g, p, e


if __name__ == '__main__':
    build()
