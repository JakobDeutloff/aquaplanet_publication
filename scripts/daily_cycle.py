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
hists_2d, hist_ccic, SW_in = load_daily_cycle_dists()
edges = np.arange(0, 25)

# %% calculate average dist
hists_2d_average = {}
for run in runs:
    hists_2d_average[run] = hists_2d[run]['hist'].sel(iwp=slice(1, None)).sum(
        ["iwp", "time"]
    ) / hists_2d[run]['size'].sum("time")

hist_ccic_sea_average_nonnorm = (
    hist_ccic["hist"].sel(iwp=slice(1, None)).sum(["time", "iwp"])
    / hist_ccic["size"].sum(["time"]))

hist_ccic_scaled = hist_ccic_sea_average_nonnorm * 0.5

# %% plot 2d hists
fig, ax1 = plt.subplots(figsize=(5, 2.5))

for run in runs:
    ax1.stairs(
        hists_2d_average[run].values,
        edges,
        label=line_labels[run],
        color=colors[run],
    )
ax1.stairs(
    hist_ccic_scaled.values,
    edges,
    label=r"CCIC / 2",
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
ax1.set_ylim([0.0005, 0.0009])
ax2.set_ylim([0, 1400])
ax2.set_ylabel("Incoming SW Radiation / W m$^{-2}$", color="grey")
ax2.set_yticks([0, 700, 1400])
ax2.tick_params(axis="y", labelcolor="grey")
# add legend
ax1.legend(frameon=False)
ax1.set_yticks([0.0005, 0.0007, 0.0009])
fig.savefig("plots/diurnal_cycle_non_normalised.pdf", bbox_inches="tight")

