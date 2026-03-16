# %%
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import (
    load_cre,
    load_definitions,
    load_iwp_distributions,
    load_lowcloud_fractions,
)

# %% load data
cre = {
    "raw": load_cre(name="raw"),
    "wetsky": load_cre(name="wetsky"),
}
cre_conn = {
    "5": load_cre(name="raw"),
    "1": load_cre(name="conn_1"),
    "20": load_cre(name="conn_20"),
}
definitions = load_definitions()
histograms = load_iwp_distributions()
lwp_fraction, cong_fraction, lc_fraction = load_lowcloud_fractions()

# %% calculate feedback values
runs = ["jed0011", "jed0022", "jed0033"]
cases = ["5", "1", "20"]

cre_folded = {}
const_iwp_folded = {}
const_cre_folded = {}

for case in cases:
    cre_folded[case] = {}
    const_iwp_folded[case] = {}
    const_cre_folded[case] = {}
    for run in runs:
        cre_folded[case][run] = cre_conn[case][run] * histograms[run].values
        const_iwp_folded[case][run] = cre_conn[case][run] * histograms["jed0011"].values
        const_cre_folded[case][run] = cre_conn[case]["jed0011"] * histograms[run].values

#  calculate integrated CRE and feedback
cre_integrated = {}
cre_const_iwp_integrated = {}
cre_const_cre_integrated = {}
feedback = {}
feedback_const_iwp = {}
feedback_const_cre = {}

for case in cases:
    cre_integrated[case] = {}
    cre_const_iwp_integrated[case] = {}
    cre_const_cre_integrated[case] = {}
    feedback[case] = {}
    feedback_const_iwp[case] = {}
    feedback_const_cre[case] = {}

    for run in runs:
        cre_integrated[case][run] = cre_folded[case][run].sum(dim="iwp")
        cre_const_iwp_integrated[case][run] = const_iwp_folded[case][run].sum(dim="iwp")
        cre_const_cre_integrated[case][run] = const_cre_folded[case][run].sum(dim="iwp")

    feedback[case]["jed0033"] = (
        cre_integrated[case]["jed0033"] - cre_integrated[case]["jed0011"]
    ) / 2
    feedback[case]["jed0022"] = (
        cre_integrated[case]["jed0022"] - cre_integrated[case]["jed0011"]
    ) / 4
    feedback_const_iwp[case]["jed0033"] = (
        cre_const_iwp_integrated[case]["jed0033"]
        - cre_const_iwp_integrated[case]["jed0011"]
    ) / 2
    feedback_const_iwp[case]["jed0022"] = (
        cre_const_iwp_integrated[case]["jed0022"]
        - cre_const_iwp_integrated[case]["jed0011"]
    ) / 4
    feedback_const_cre[case]["jed0033"] = (
        cre_const_cre_integrated[case]["jed0033"]
        - cre_const_cre_integrated[case]["jed0011"]
    ) / 2
    feedback_const_cre[case]["jed0022"] = (
        cre_const_cre_integrated[case]["jed0022"]
        - cre_const_cre_integrated[case]["jed0011"]
    ) / 4


# %% plot feedback values
parts = ["lw", "sw", "net"]
feedbacks = {
    "cre": feedback_const_iwp,
    "iwp": feedback_const_cre,
    "Total Feedback": feedback,
}
offsets = {
    "lw": 0,
    "sw": 1,
    "net": 2,
}
colors = {
    "lw": definitions[5],
    "sw": definitions[4],
    "net": definitions[6],
}
markers = {
    "5": "o",
    "1": "s",
    "20": "^",
}
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True, sharex=True)
for i, f_name in enumerate(feedbacks.keys()):
    for case in cases:
        for part in parts:
            axes[i].scatter(
                offsets[part],
                feedbacks[f_name][case]["jed0022"][part],
                color=colors[part],
                marker=markers[case],
                facecolor="none",
            )

for ax in axes:
    ax.set_xticks([0, 1, 2], ["LW", "SW", "Net"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)

axes[0].set_ylabel(r"$F_{\mathrm{C}}$ / W m$^{-2}$ K$^{-1}$")
axes[1].set_ylabel(r"$F_{\mathrm{P}}$ / W m$^{-2}$ K$^{-1}$")
axes[2].set_ylabel(r"$F$ / W m$^{-2}$ K$^{-1}$")
handles = [
    plt.Line2D([0], [0], color="grey", linestyle="", marker=markers["1"], label="1 %"),
    plt.Line2D([0], [0], color="grey", linestyle="", marker=markers["5"], label="5 %"),
    plt.Line2D(
        [0], [0], color="grey", linestyle="", marker=markers["20"], label="20 %"
    ),
]
fig.legend(
    handles,
    [handle.get_label() for handle in handles],
    bbox_to_anchor=(0.7, 0),
    ncol=3,
    frameon=False,
)

# label all plots 
for j, ax in enumerate(axes):
    ax.text(
        0.1,
        1.05,
        chr(97 + j),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

fig.savefig("plots/feedback_sensitivity.pdf", bbox_inches="tight")

# %% plot connected LWP fraction
fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
for run in runs:
    axes[0].plot(
        lwp_fraction[run]["iwp_bins"],
        lwp_fraction[run],
        color=definitions[2][run],
        label=definitions[3][run],
    )

    axes[1].plot(
        lc_fraction[run]["iwp_bins"],
        lc_fraction[run],
        color=definitions[2][run],
        label=definitions[3][run],
    )

    axes[2].plot(
        cong_fraction[run]["iwp_bins"],
        cong_fraction[run],
        color=definitions[2][run],
        label=definitions[3][run],
    )
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e-1)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel('Liquid CF')
axes[0].set_ylim(0.42, 0.57)
axes[1].set_ylabel('Low CF')
axes[1].set_ylim(0.42, 0.57)
axes[2].set_ylabel('Congestus CF')
axes[2].set_ylim(0, 0.15)
axes[2].set_xlabel('$I$ / kg m$^{-2}$')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    bbox_to_anchor=(0.7, 0),
    ncol=3,
    frameon=False,
)
# label all axes
for j, ax in enumerate(axes):
    ax.text(
        0.05,
        1.05,
        chr(97 + j),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )
fig.savefig("plots/lcf_comparison.pdf", bbox_inches="tight")

#%% plot plot SW cre diff for wetsky
fig, ax = plt.subplots(figsize=(7, 3))
linestyles = definitions[-1]
ax.axhline(0, color="k", linewidth=0.5)
ax.plot(
    cre["wetsky"]["jed0022"]["iwp"],
    cre["wetsky"]["jed0022"]["sw"] - cre["wetsky"]["jed0011"]["sw"],
    color=colors["sw"],
    linestyle=linestyles["jed0022"],
    label='+4 K'
)
ax.plot(
    cre["wetsky"]["jed0033"]["iwp"],
    cre["wetsky"]["jed0033"]["sw"] - cre["wetsky"]["jed0011"]["sw"],
    color=colors["sw"],
    linestyle=linestyles["jed0033"],
    label='+2 K'
)

ax.spines[["top", "right"]].set_visible(False)
ax.set_xscale("log")
ax.set_xlim(1e-4, 10)
ax.set_ylim(-3, 27)
ax.set_yticks([0, 5, 25])
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel(r"$\Delta C_{\mathrm{SW}}(I)$ / W m$^{-2}$")
ax.legend(frameon=False)
fig.savefig("plots/sw_cre_comparison.pdf", bbox_inches="tight")

# %%
