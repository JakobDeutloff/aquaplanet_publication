# %%
import matplotlib.pyplot as plt
from src.read_data import load_random_datasets, load_definitions
import numpy as np

# %%
datasets = load_random_datasets()
definitions = load_definitions()
runs = ["jed0011", "jed0033", "jed0022"]
# %% find height level close to 500 hPa
height_500 = {}
for run in runs:
   diff =  (datasets[run]['pfull'].mean('index')/1e2) - 500
   height_500[run] = np.abs(diff).argmin().values

# %% select vertical velocity at 500 hPa
w_500 = {}
for run in runs:
   w_500[run] = datasets[run]['wa'].isel(height_2=height_500[run])

# %% calculate histograms 
hists = {}
edges = np.arange(-5, 14, 0.5)
for run in runs:
    hist, bin_edges = np.histogram(
        w_500[run].values.flatten(),
        bins=edges,
        density=False,
    )
    hist = hist / hist.sum()
    hists[run] = hist
# %% plot disributions of vertical velocity at 500 hPa
fig, ax = plt.subplots(figsize=(5, 2.5))
line_labels = definitions[3]
colors = definitions[2]
for run in runs:
    ax.stairs(
        hists[run],
        edges,
        label=line_labels[run],
        color=colors[run],
    )
ax.set_yscale('log')
ax.legend(frameon=False)
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel(r'$\omega_{\mathrm{500}}$ / m s$^{-1}$')
ax.set_ylabel(r'$P(\omega_{\mathrm{500}}$)')
fig.savefig('plots/vertical_velocity_500hPa_distribution.pdf', bbox_inches='tight')

# %%
