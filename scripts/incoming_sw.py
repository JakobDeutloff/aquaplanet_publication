# %%
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_definitions, load_sw_metrics

# %% load CRE data
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
sw_down_binned, rad_time_binned, lat_binned = load_sw_metrics()


# %% plot mean time and SW down
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True)

for run in runs:
    sw_down_binned[run].sel(iwp_points=slice(1e-4, 20)).plot(
        ax=axes[0], label=line_labels[run], color=colors[run]
    )
    rad_time_binned[run].sel(iwp_points=slice(1e-4, 20)).plot(
        ax=axes[1], label=line_labels[run], color=colors[run]
    )
    lat_binned[run].sel(iwp_points=slice(1e-4, 20)).plot(
        ax=axes[2], label=line_labels[run], color=colors[run]
    )

axes[0].set_ylabel("SW down / W m$^{-2}$")
axes[1].set_ylabel("Time Difference to Noon / h")
axes[2].set_ylabel("Distance to equator / deg")
axes[0].set_xscale("log")
for ax in axes:
    ax.set_xlabel("$I$ / $kg m^{-2}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 20])

axes[1].invert_yaxis()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.1),
    frameon=False,
)
# add letters 
for ax, letter in zip(axes, ["a", "b", "c"]):
    ax.text(
        0.03,
        1,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
axes[2].invert_yaxis()
fig.tight_layout()
fig.savefig("plots/sw_incoming.pdf", bbox_inches="tight")

# %%
