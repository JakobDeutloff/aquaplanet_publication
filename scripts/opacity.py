# %%
import matplotlib.pyplot as plt
import pickle
from src.read_data import load_definitions, load_emissivity_albedo
import xarray as xr

# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
emissivity, albedo = load_emissivity_albedo()
# %% plot albedo and emissivity
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
for run in ["jed0022", "jed0033", "jed0011"]:
    axes[0].plot(
        albedo[run].iwp_bins,
        albedo[run],
        color=colors[run],
        label=line_labels[run],
        linestyle=linestyles[run],
    )
    axes[1].plot(
        emissivity[run].iwp_bins,
        emissivity[run],
        color=colors[run],
        label=line_labels[run],
        linestyle=linestyles[run],
    )

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("$I$ / kg m$^{-2}$")

axes[1].set_ylabel("Emissivity")
axes[0].set_ylabel("Albedo")

# put letters
for i, ax in enumerate(axes):
    ax.text(
        0.1,
        0.97,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.65, 0), ncol=3, ncols=3, frameon=False)
fig.savefig('plots/albedo_emissivity.pdf', bbox_inches='tight')
# %%
