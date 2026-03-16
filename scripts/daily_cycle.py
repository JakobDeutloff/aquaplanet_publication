# %%
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_definitions, load_daily_cycle_dists
import xarray as xr
import glob
import re

# %% load CRE data
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
iwp_bins = np.logspace(-4, np.log10(40), 51)
T_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
hists, hist_ccic, SW_in = load_daily_cycle_dists()
edges = np.arange(0, 25)
hists = {run: hists[run]["__xarray_dataarray_variable__"] for run in runs}

# %% load 2d hists
temp_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
hists_2d = {}
for run in runs:
    hists_2d[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/daily_cycle_hist_2d.nc"
    ).sum("time")


# %% average histograms
hists_average = {}
edges = np.arange(0, 25, 1)
for run in runs:
    hists_average[run] = (hists[run].sum("day") / hists[run].sum()).values

# %% average ccic
hist_ccic_sea_average = (
    hist_ccic["hist"].sel(iwp=slice(1, None)).sum(["time", "iwp"])
    / hist_ccic["hist"]
    .sel(iwp=slice(1, None))
    .sum(["time", "local_time", "iwp"])
    .sum()
)

# %% plot 1 kg m^-2
fig, ax1 = plt.subplots(figsize=(5, 2.5))

for run in runs:
    ax1.stairs(
        hists_average[run],
        edges,
        label=line_labels[run],
        color=colors[run],
    )
ax1.stairs(
    hist_ccic_sea_average,
    edges,
    label="CCIC",
    color="black",
)
ax2 = ax1.twinx()
SW_in.plot(ax=ax2, color="grey", linewidth=3, alpha=0.5)

for ax in [ax1, ax2]:
    ax.set_xlim([0.1, 23.9])
    ax.spines[["top"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_xlabel("Local Time / h")

ax1.set_ylabel("P($I$ > 1 kg m$^{-2}$)")
ax1.set_ylim([0.0325, 0.054])
ax1.set_yticks([0.04, 0.05])
ax2.set_ylim([0, 1400])
ax2.set_ylabel("Incoming SW Radiation / W m$^{-2}$", color="grey")
ax2.set_yticks([0, 700, 1400])
ax2.tick_params(axis="y", labelcolor="grey")
# add legend
ax1.legend(frameon=False)

fig.savefig("plots/diurnal_cycle_1.pdf", bbox_inches="tight")

# %% 
