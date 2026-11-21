"""Plot editing and feedback costs per assigned Blender task; final grading is separate."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager
from prepare import R, read

zh = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(zh)
plt.rcParams.update(
    {
        'font.family': font_manager.FontProperties(fname=zh).get_name(),
        'axes.unicode_minus': False,
        'savefig.facecolor': 'white',
        'axes.facecolor': 'white',
    }
)


def main():
    g = read(R / 'metrics/findings.json')['groups']
    fig, axs = plt.subplots(1, 2, figsize=(10.3, 3.8), sharex=True)
    arms = ['F0', 'F0R', 'F1R']
    labels = ['F0', 'F0 ＋规则', 'F1 ＋规则']
    peak = max(
        (
            g[f'3dcodeedit/{s}/{a}']['editor_known_tokens']
            + g[f'3dcodeedit/{s}/{a}']['reviewer_known_tokens']
        )
        / g[f'3dcodeedit/{s}/{a}']['assigned']
        / 1000
        for s in ['simple', 'general']
        for a in arms
    )
    for ax, t, s in zip(
        axs, ['简单候选（8 题）', '一般任务（12 题）'], ['simple', 'general']
    ):
        for i, a in enumerate(arms):
            v = g[f'3dcodeedit/{s}/{a}']
            n = v['assigned']
            x = v['editor_known_tokens'] / n / 1000
            y = v['reviewer_known_tokens'] / n / 1000
            ax.barh(
                i,
                x,
                height=0.44,
                color='#176456',
                label='编辑' if i == 0 else None,
            )
            ax.barh(
                i,
                y,
                left=x,
                height=0.44,
                color='#e0aa63',
                label='反馈审查' if i == 0 else None,
            )
            ax.text(
                x + y + peak * 0.015,
                i,
                f'{x + y:.1f}',
                va='center',
                fontsize=9,
            )
        ax.set(
            title=t,
            xlim=(0, peak * 1.16),
            yticks=[0, 1, 2],
            yticklabels=labels,
            ylim=(2.65, -0.62),
            xlabel='每题用量（千 tokens）',
        )
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis='x', alpha=0.14)
        ax.set_axisbelow(True)
    fig.subplots_adjust(
        left=0.105, right=0.98, top=0.79, bottom=0.24, wspace=0.32
    )
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )
    fig.suptitle(
        '3DCodeEdit：交付一题，编辑与反馈各用了多少？',
        fontsize=14,
        weight='bold',
    )
    fig.savefig(R / 'comparisons/code-tokens.png', dpi=155)
    plt.close(fig)


if __name__ == '__main__':
    main()
