# %%
import matplotlib.pyplot as plt
from src.read_data import load_definitions, load_hr_components

# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = load_definitions()
f_conv, mean_hr, mean_rho = load_hr_components()
# %% plot convergence of net flux
fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharey=True)

for run in runs:

    axes[0].plot(
        mean_hr[run],
        mean_hr[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[2].plot(
        mean_rho[run],
        mean_rho[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[1].plot(
        f_conv[run],
        f_conv[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )



axes[0].invert_yaxis()
axes[0].set_ylabel("Temperature / K")
axes[1].set_xlabel("Net flux divergence / W m$^{-3}$")
axes[2].set_xlabel("Air Density / kg m$^{-3}$")
axes[0].set_xlabel("Heating rate / K day$^{-1}$")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xticks([-0.01, -0.005, 0])

handles, names = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    names,
    loc="center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
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
fig.tight_layout()
fig.savefig("plots/flux_conv.pdf", bbox_inches="tight")
# %%
