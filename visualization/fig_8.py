import numpy as np
import matplotlib.pyplot as plt
#colors  = ['#84A5C7', '#f18484', '#f8ad64']
import numpy as np
import matplotlib.pyplot as plt
from imageio import formats
from matplotlib.patches import Patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- 三个数据集原始数据（Bus 移除 50%） ---
data_dict = {
    'Bus': {
        'mask_ratios': ['10%', '20%', '30%'],
        'values': np.array([
            [ 2.9636,  6.6383, 10.6282],  # GWNET
            [ 3.6744,  8.1850, 11.2078],  # DGCRN
            [ 2.3413,  6.0928,  7.8712],  # MM-dy
        ]),
        'yticks': np.array([0, -5, -10, -15])
    },
    'Metro': {
        'mask_ratios': ['10%', '20%', '30%'],
        'values': np.array([
            [15.3183, 32.0486, 47.3049],
            [12.6031, 23.1945, 30.1183],
            [ 5.0407, 12.0429, 19.3584],
        ]),
        'yticks': np.array([0, -10, -20, -30, -40, -50, -60])
    },
    'Taxi': {
        'mask_ratios': ['10%', '20%', '30%'],
        'values': np.array([
            [0.9810, 2.5966, 3.4398],
            [0.9130, 1.5379, 2.1778],
            [0.5020, 1.0426, 1.5259],
        ]),
        'yticks': np.array([0, -1, -2, -3, -4])
    },
}

models  = ['GWNET', 'DGCRN', 'MM-dyGNN']
colors  = ['#84A5C7', '#f18484', '#f8ad64']
hatches = ['/', '\\', 'x']

# 创建 1x3 并排子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False,constrained_layout=True)

for ax, (dataset, info) in zip(axes, data_dict.items()):
    ratios = info['mask_ratios']
    vals   = info['values']
    inv    = -vals  # 向下柱状

    n_models, n_masks = inv.shape
    x = np.arange(n_masks)
    width = 0.8 / n_models
    offsets = (np.arange(n_models) - (n_models-1)/2) * width

    # 绘制每个模型的柱状
    for i, (m, c, h) in enumerate(zip(models, colors, hatches)):
        xi = x + offsets[i]
        yi = inv[i]
        bars = ax.bar(xi, yi, width=width, color=c, hatch=h, edgecolor='black')
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height - 0.2,
                f'{-height:.2f}',
                ha='center', va='top', fontsize=9
            )

    ax.set_title(dataset, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ratios, fontsize=11)
    ax.set_xlabel('Spatial Node Mask Ratio', fontsize=12)
    ax.set_yticks(info['yticks'])
    ax.set_ylim(info['yticks'].min(), 0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# 在第一个子图上设置 Y 轴标签
axes[0].set_ylabel('Performance Drop', fontsize=12)

# 统一图例，扩大范围并完整显示 hatches
handles = [
    Patch(facecolor=c, edgecolor='black', hatch=h, label=m)
    for m, c, h in zip(models, colors, hatches)
]
fig.legend(
    handles=handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.03),
    ncol=3,
    fontsize=12,
    title_fontsize=14,
    frameon=False,
    handlelength=3,
    handleheight=2,
    borderpad=1
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
# 保存为 PNG
plt.savefig('performance_drop.png',
            dpi=300,
            bbox_inches='tight')

# 保存为 SVG
plt.savefig('performance_drop.svg',
            dpi=300,
            bbox_inches='tight')

# 保存为 PDF
plt.savefig('performance_drop.pdf',
            dpi=300,
            bbox_inches='tight')
plt.show()

