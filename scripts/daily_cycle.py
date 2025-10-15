# %%
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_definitions, load_daily_cycle_dists

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

# %% average histograms
hists_average = {}
edges = np.arange(0, 25, 1)
for run in runs:
    hists_average[run] = (hists[run].sum("day") / hists['jed0011'].sum())[
        "__xarray_dataarray_variable__"
    ].values

# %% plot 1 kg m^-2
fig, ax1 = plt.subplots(figsize=(5, 2.5))

for run in runs:
    ax1.stairs(
        hists_average[run], edges, label=line_labels[run], color=colors[run],
    )
ax2 = ax1.twinx()
SW_in.plot(
    ax=ax2,
    color="grey",
    linewidth=3,
    alpha=0.5)

for ax in [ax1, ax2]:
    ax.set_xlim([0.1, 23.9])
    ax.spines[["top"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_xlabel("Local Time / h")

ax1.set_ylabel("P($I$ > 1 kg m$^{-2}$)")
ax1.set_ylim([0.03, 0.051])
ax1.set_yticks([0.03, 0.04, 0.05])
ax2.set_ylim([0, 1400])
ax2.set_ylabel("Incoming SW Radiation / W m$^{-2}$", color='grey')
ax2.set_yticks([0, 700, 1400])
ax2.tick_params(axis='y', labelcolor='grey')
# add legend
ax1.legend(frameon=False)

fig.savefig("plots/diurnal_cycle_1.pdf", bbox_inches="tight")


# %% calculate difference between peaks 
diffs = {}
temp_delats = {
    'jed0022': 4,
    'jed0033': 2,
}
for run in runs:
    max_morning = hists_average[run].max()
    max_aternoon = hists_average[run][12:20].max()
    diffs[run] = (max_morning - max_aternoon)

for run in runs[1:]:
    print(f"Increase in diurnal cycle for {line_labels[run]} compared to control by {(((diffs[run]-diffs['jed0011']) / diffs['jed0011'])*100):.0f}%")

# %%
