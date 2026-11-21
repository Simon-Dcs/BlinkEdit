"""A live, read-only view of the existing receipts; never triggers an edit or consumes tokens."""

from datetime import datetime
from pathlib import Path
import time
from design import R, read


def refresh():
    r = R
    manifest = read(r / 'manifest.json')
    budget = read(r / 'metrics/formal-budget.json')
    completed = judged = pairs = 0
    latest = 0
    known = 0
    active = 0
    fail = 0
    for key in manifest['formal']['code']:
        for a in manifest['arms']:
            ds = sorted(
                (r / '3dcodeedit/formal' / key / a).glob('turn*/outcome.json')
            )
            if not ds:
                continue
            v = read(ds[-1])
            if not v.get('error') and (
                v.get('accept') or v['turn'] >= 1 + manifest['max_revisions']
            ):
                completed += 1
                judged += int(ds[-1].with_name('score.json').is_file())
    for pair in manifest['paired_trials']:
        ds = [
            r
            / 'edithero/formal'
            / pair['case']
            / f"p{pair['i']}r{pair['rep']}-{method}"
            / f"turn{pair['turn']:02}"
            for method in pair['order']
        ]
        if all(
            (d / 'pair-score.json').exists()
            and not any(
                read(p).get('status') == 'failed'
                for p in d.rglob('record.json')
            )
            for d in ds
        ):
            pairs += 1
    for d in budget['invocations']:
        ps = list(Path(d).rglob('record.json'))
        if not ps:
            active += 1
        for p in ps:
            v = read(p)
            usage = v.get('usage') or {}
            known += usage.get('input_tokens', 0) + usage.get(
                'output_tokens', 0
            )
            fail += int(not usage)
            latest = max(latest, p.stat().st_mtime)

    def display(t):
        return datetime.fromtimestamp(t).strftime('%m-%d %H:%M:%S')

    text = f"""# 本轮实验进度

更新时间：{display(time.time())}（北京时间）。最近一次请求回执：{display(latest)}。

| 阶段 | 已完成 |
|---|---:|
| EditHero：两个编辑方法同状态成对运行并评分 | {pairs} / 24 对 |
| 3DCodeEdit：程序编辑和过程反馈结束 | {completed} / 72 份 |
| 3DCodeEdit：最终产物逐项盲评结束 | {judged} / 72 份 |

已登记 {len(budget['invocations'])} 次请求；有回执的用量 {known:,} tokens。{active} 个请求已登记、尚未收到回执，{fail} 份回执没有用量。

[实验报告](../../../../edit4.md)
"""
    dest = r / 'report/progress.md'
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(text)
    tmp.replace(dest)
    return (pairs == 24 and judged == 72) or time.time() - budget[
        'start'
    ] >= read(r / 'metrics/formal-limits.json')['seconds']


if __name__ == '__main__':
    while not refresh():
        time.sleep(120)
