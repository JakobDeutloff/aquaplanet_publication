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
hists, SW_in = load_daily_cycle_dists()
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


# %% load ccic data
files = glob.glob(
    "/work/bm1183/m301049/ccic_daily_cycle/*/ccic_cpcir_daily_cycle_distribution_2d*.nc"
)
files_sea = [f for f in files if re.search(r"2d_sea_\d{4}\.nc$", f)]
hist_ccic_sea = xr.open_mfdataset(files_sea).load()

# %% average histograms
hists_average = {}
edges = np.arange(0, 25, 1)
for run in runs:
    hists_average[run] = (hists[run].sum("day") / hists[run].sum()).values

# %% average ccic
hist_ccic_sea_average = (
    hist_ccic_sea["hist"].sel(iwp=slice(1, None)).sum(["time", "iwp"])
    / hist_ccic_sea["hist"]
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


# %% calculate difference between peaks
diffs = {}
temp_delats = {
    "jed0022": 4,
    "jed0033": 2,
}
for run in runs:
    max_morning = hists_average[run].max()
    max_aternoon = hists_average[run][12:20].max()
    diffs[run] = max_morning - max_aternoon

for run in runs[1:]:
    print(
        f"Increase in diurnal cycle for {line_labels[run]} compared to control by {(((diffs[run]-diffs['jed0011']) / diffs['jed0011'] / temp_delats[run])*100):.0f}% per K"
    )


# %% plot change in histogram only for iwp < 1
hist_thin = {}
hist_intermediate = {}
for run in runs:
    hist_thin[run] = hists_2d[run]["hist"].sel(iwp=slice(1e-1, 1)).sum(
        ["iwp"]
    ) / hists_2d[run]["hist"].sel(iwp=slice(1e-1, 1)).sum(["local_time", "iwp"])
    hist_intermediate[run] = hists_2d[run]["hist"].sel(iwp=slice(5e-1, 1)).sum(
        ["iwp"]
    ) / hists_2d[run]["hist"].sel(iwp=slice(5e-1, 1)).sum(["local_time", "iwp"])

hist_thin_rel = {}
hist_intermediate_rel = {}
hist_thick_rel = {}
for run in runs[1:]:
    hist_thin_rel[run] = ((hist_thin[run] - hist_thin[runs[0]]) * 100) / (
        temp_delta[run] * hist_thin[runs[0]]
    )  # % / K
    hist_intermediate_rel[run] = ((hist_intermediate[run] - hist_intermediate[runs[0]]) * 100) / (
        temp_delta[run] * hist_intermediate[runs[0]]
    )  # % / K
    hist_thick_rel[run] = ((hists_average[run] - hists_average[runs[0]]) * 100) / (
        temp_delta[run] * hists_average[runs[0]]
    )  # % / K

# %%
fig, axes = plt.subplots(3, 1, figsize=(5, 7), sharex=True)

for run in runs[1:]:
    axes[2].stairs(
        hist_thin_rel[run],
        edges,
        label=line_labels[run],
        color=colors[run],
    )
    axes[1].stairs(
        hist_intermediate_rel[run],
        edges,
        label=line_labels[run],
        color=colors[run],
    )
    axes[0].stairs(
        hist_thick_rel[run],
        edges,
        label=line_labels[run],
        color=colors[run],
    )
for ax in axes:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlim([0.1, 23.9])
    ax.set_xticks([6, 12, 18])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(r"$\dfrac{\mathrm{d}P}{P\mathrm{d}T}$ / % K$^{-1}$")
axes[2].set_xlabel("Local Time / h")
axes[2].set_title("0.1 kg m$^{-2}$ < $I$ < 0.5 kg m$^{-2}$")
axes[1].set_title("0.5 kg m$^{-2}$ < $I$ < 1 kg m$^{-2}$")
axes[0].set_title("$I$ > 1 kg m$^{-2}$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.7, 0), ncol=2, frameon=False)
fig.savefig("plots/relative_diurnal_cycle_change.pdf", bbox_inches="tight")


# %% 
