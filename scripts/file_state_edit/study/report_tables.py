"""Format the terminal evidence tables. The interpretation is written against these receipts."""

from analysis import analyse
from prepare import R, read
from pathlib import Path
import statistics as st


def render():
    groups, paired, cases = analyse()
    table = [
        '| 样本、方法 | 首稿反馈接受 | 额外修正 | 最终盲评全通过 | 编辑 tokens | 反馈审查 tokens | 编辑＋反馈 tokens | 最终评分 tokens | 编辑至反馈结束（秒） | 最终评分模型耗时（秒） |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    legend = {'F0': 'F0', 'F0R': 'F0＋规则', 'F1R': 'F1＋规则'}
    for g, title in [('simple', '简单候选'), ('general', '一般任务')]:
        for a in legend:
            z = groups[f'3dcodeedit/{g}/{a}']
            tok = z['editor_known_tokens']
            rev = z['reviewer_known_tokens']
            table.append(
                f"| {title} · {legend[a]} | {z['first_accepted']}/{z['assigned']} | {z['repairs']} | {z['passed']}/{z['judged']} | {tok:,} | {rev:,} | {tok + rev:,} | {z['verifier_known_tokens']:,} | {z['elapsed_s']:,.2f} | {z['verifier_s']:,.2f} |"
            )
    path = R / 'report/code-table.md'
    path.write_text('\n'.join(table) + '\n')
    print('\n'.join(table))


if __name__ == '__main__':
    render()
