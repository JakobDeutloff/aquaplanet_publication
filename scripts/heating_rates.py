# %%
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_definitions, load_hr_and_cf
# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
hrs_binned_net, hrs_binned_lw, hrs_binned_sw, cf_binned = load_hr_and_cf()
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2


# %% plot
fig, axes  = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
cmap = "seismic"
net = axes[0].pcolormesh(
    iwp_points,
    hrs_binned_net['jed0011']['temp'],
    hrs_binned_net['jed0011'].T,
    cmap=cmap,
    vmin=-4,
    vmax=4,
    rasterized=True
)

diff = axes[1].pcolormesh(
    iwp_points,
    hrs_binned_net['jed0033']['temp'],
    (hrs_binned_net['jed0033'] - hrs_binned_net['jed0011']).T,
    cmap=cmap,
    vmin=-1.5,
    vmax=1.5,
    rasterized=True
)

axes[2].pcolormesh(
    iwp_points,
    hrs_binned_net['jed0022']['temp'],
    (hrs_binned_net['jed0022'] - hrs_binned_net['jed0011']).T,
    cmap=cmap,
    vmin=-1.5,
    vmax=1.5,
    rasterized=True
)

for i, run in enumerate(runs):
    contour = axes[i].contour(
        iwp_points,
        cf_binned[run]['temp'],
        cf_binned[run].T,
        colors="k",
        levels=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    axes[i].clabel(contour, inline=True, fontsize=8, fmt="%1.1f")
    axes[i].set_xscale("log")
    axes[i].set_xlim(iwp_bins[0], iwp_bins[-1])
    axes[i].invert_yaxis()
    axes[i].set_ylim([260, 190])
    axes[i].set_xlabel("$I$ / kg m$^{-2}$")

axes[0].set_ylabel("Temperature / K")
axes[0].set_yticks([260, 230, 210, 190])
axes[0].set_title("Control")
axes[1].set_title("+2 K - Control")
axes[2].set_title("+4 K - Control")


cbar_height = 0.03
cbar_bottom = -0.09  # vertical position
cbar_left1 = 0.13   # left for first colorbar
cbar_left2 = 0.4   # left for second colorbar

cax1 = fig.add_axes([cbar_left1, cbar_bottom, 0.22, cbar_height])
cax2 = fig.add_axes([cbar_left2, cbar_bottom, 0.5, cbar_height])


cb_net = fig.colorbar(mappable=net, cax=cax1, orientation="horizontal")
cb_net.set_label("$Q$ / K day$^{-1}$")

cb_diff = fig.colorbar(mappable=diff, cax=cax2, orientation="horizontal")
cb_diff.set_label(r"$\Delta Q $ / K day$^{-1}$")

# add letters
for i, ax in enumerate(axes):
    ax.text(
        0.03,
        1.05,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

fig.savefig("plots/heating_rates.pdf", bbox_inches="tight", dpi=300)
# %%
