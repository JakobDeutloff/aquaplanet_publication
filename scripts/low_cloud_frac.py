# %%
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_definitions, load_lc_fractions

# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
lc_binned, lc_binned_raw = load_lc_fractions()

# %% plot lc fraction
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
for run in runs:
    axes[0].plot(
        iwp_points,
        lc_binned[run],
        label=line_labels[run],
        color=colors[run],
    )
    axes[1].plot(
        iwp_points,
        lc_binned_raw[run],
        color=colors[run],
    )

for ax in axes:
    ax.set_xlabel("$I$ / $kg m^{-2}$")
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Liquid Cloud Fraction")
axes[0].set_title("With Tuning Factor")
axes[1].set_title("Without Tuning Factor")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(runs),
    bbox_to_anchor=(0.5, -0.15),
    frameon=False,
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

fig.savefig("plots/lc_frac.pdf", bbox_inches="tight")

# %%
