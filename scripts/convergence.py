# %%
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_definitions, load_iwp_distributions, load_iwp_distributions_30, load_iwp_distributions_60

# %% load histograms 
runs = ['jed0011', 'jed0022', 'jed0033']

definitions = load_definitions()
colors = definitions[2]
hists_full = load_iwp_distributions()
hists_30 = load_iwp_distributions_30()
hists_60 = load_iwp_distributions_60()


# %% plot histograms
iwp_bins = np.logspace(-4, np.log10(40), 51)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

axes[0].stairs(
    hists_30['jed0033'] - hists_30['jed0011'],
    iwp_bins,
    color='blue',
)
axes[0].stairs(
    hists_60['jed0033'] - hists_60['jed0011'],
    iwp_bins,
    color='red')
axes[0].stairs(
    hists_full['jed0033'] - hists_full['jed0011'],
    iwp_bins,
    color='k',
)

axes[1].stairs(
    hists_30['jed0022'] - hists_30['jed0011'],
    iwp_bins,
    label='1 month',
    color='blue',
)
axes[1].stairs(
    hists_60['jed0022'] - hists_60['jed0011'],
    iwp_bins,
    label='2 months',
    color='red')
axes[1].stairs(
    hists_full['jed0022'] - hists_full['jed0011'],
    iwp_bins,
    label='3 months',
    color='k',
)
for ax in axes:
    ax.set_xscale('log')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(1e-4, 40)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('$I$ / kg m$^{-2}$')

axes[0].set_ylabel(r'$\Delta P(I)$')
axes[0].set_title('+2 K')
axes[1].set_title('+4 K')

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1),
    frameon=False
)
# add letters 
for ax, letter in zip(axes, ["a", "b"]):
    ax.text(
        0.03,
        1,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
fig.savefig('plots/sup_conv.pdf', bbox_inches='tight')

# %%
