from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False

fig, ax = plt.subplots(figsize=(3.5, 2.5))

for i, (mode, data) in enumerate(values.items()):
    yerr = ci[mode] if ci else None
    ax.errorbar(k_labels, data, yerr=yerr,
                marker='o', linestyle='-', linewidth=1.0,
                markersize=4, capsize=3,
                label=mode, color=palette(i))

ax.set_xlabel(r'Top-$k$ threshold')
ax.set_ylabel('MAE ↓')
ax.set_ylim(2, 20)                              # full range for fair comparison
ax.grid(axis='y', ls='--', lw=0.4, alpha=.7)

ax.legend(frameon=False, ncol=3,
          bbox_to_anchor=(0.5, 1.12), loc='upper center')

fig.tight_layout()
fig.savefig(outdir / 'topk_comparison.pdf', bbox_inches='tight')
fig.savefig(outdir / 'topk_comparison.png', dpi=300, bbox_inches='tight')

# ----------------------------------------------------------------------
# 2) Three aligned sub-plots (alternative figure)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharex=True)

for ax, (mode, data) in zip(axes, values.items()):
    idx = list(values).index(mode)
    yerr = ci[mode] if ci else None
    ax.errorbar(k_labels, data, yerr=yerr,
                marker='o', linestyle='-', linewidth=1.0,
                markersize=4, capsize=3,
                color=palette(idx))

    ax.set_title(mode)
    ax.set_ylim(*y_ranges[mode])
    ax.grid(axis='y', ls='--', lw=0.4, alpha=.7)
    ax.set_xlabel(r'$k$')
    if ax is axes[0]:
        ax.set_ylabel('MAE ↓')

fig.tight_layout(w_pad=1.0)
fig.savefig(outdir / 'topk_modes.pdf', bbox_inches='tight')
fig.savefig(outdir / 'topk_modes.png', dpi=300, bbox_inches='tight')

# ----------------------------------------------------------------------
# 3) Individual SVGs (supplementary / slides)
# ----------------------------------------------------------------------
for idx, (mode, data) in enumerate(values.items()):
    fig_i, ax_i = plt.subplots(figsize=(3.0, 2.2))
    yerr = ci[mode] if ci else None
    ax_i.errorbar(k_labels, data, yerr=yerr,
                  marker='o', linestyle='-', linewidth=1.0,
                  markersize=4, capsize=3,
                  color=palette(idx), label=mode)

    ax_i.set_xlabel(r'$k$')
    ax_i.set_ylabel('MAE')
    ax_i.set_ylim(*y_ranges[mode])
    ax_i.grid(axis='y', ls='--', lw=0.4, alpha=.7)

    fig_i.tight_layout()
    fig_i.savefig(outdir / f'topk_{mode.lower()}.svg',format='svg')
    plt.close(fig_i)         # free memory