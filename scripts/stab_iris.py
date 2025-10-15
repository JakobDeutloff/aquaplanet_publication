# %%
import matplotlib.pyplot as plt
from src.read_data import load_definitions, load_vgrid, load_stab_iris, load_t_profiles

# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = load_definitions()
linestyles = {
    "jed0011": "-",
    "jed0022": "-",
    "jed0033": "-",
    "jed2224": "--",}
colors["jed2224"] = "k"
line_labels["jed2224"] = "+4 K const. O3"
v_grid = load_vgrid()
hrs, stab, subs, conv, subs_cont, conv_cont = load_stab_iris()
t_profile = load_t_profiles()

# %% plot results in /m
fig, axes = plt.subplots(1, 4, figsize=(10, 4), sharey=True)
plot_const_hr = True
for run in runs:
    axes[0].plot(
        hrs[run]["net_hr"],
        hrs[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[0].set_xlabel("Heating Rate / K day$^{-1}$")
    axes[1].plot(
        stab[run],
        stab[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[1].set_xlabel("Stability / K m$^{-1}$")
    axes[2].plot(
        subs[run],
        subs[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[2].set_xlabel("Subsidence / m day$^{-1}$")
    axes[3].plot(
        conv[run],
        conv[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[3].set_xlabel("Convergence /  day$^{-1}$")
# const hr
axes[2].plot(
    subs_cont,
    subs_cont["temp"],
    label=line_labels[run],
    color=colors[run],
    linestyle="--",
)
axes[3].plot(
    conv_cont,
    conv_cont["temp"],
    color=colors[run],
    linestyle="--",
    label="+4 K Constant HR",
)

axes[0].set_ylim([260, 200])
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Temperature / K")
axes[0].set_yticks([260, 230, 215, 200])
handles, names = axes[-1].get_legend_handles_labels()
fig.legend(
    handles,
    names,
    loc="center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
    frameon=False,
)
# add letters
for i, ax in enumerate(axes):
    ax.text(
        0.03,
        1,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

fig.tight_layout()
fig.savefig("plots/stab_iris_profiles.pdf", bbox_inches="tight")


# %% plot temperature profiles in clearsky 
fig, ax = plt.subplots(figsize=(4, 6))
for run in [*runs, "jed2224"]:
    ax.plot(
        t_profile[run],
        v_grid["zg"]/1e3,
        label=line_labels[run],
        color=colors[run],
        linestyle=linestyles[run],
    )

ax.set_ylim([10, 20])
ax.set_xlim([190, 250])
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Temperature / K")
ax.set_ylabel("Height / km")
ax.legend(
    ncol=1,
    fontsize=10,
)
fig.savefig("plots/t_profile_clearsky.pdf", bbox_inches="tight")

# %%
