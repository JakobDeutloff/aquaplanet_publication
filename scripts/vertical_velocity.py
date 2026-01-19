# %%
import matplotlib.pyplot as plt
from src.read_data import load_random_datasets, load_definitions
import numpy as np
import xarray as xr
from scipy.stats import skew

# %%
definitions = load_definitions()
# %% load raw icon data 
ds_icon = xr.open_dataset('/work/bm1183/m301049/icon_hcap_data/publication/vertical_vel/vel_icon_2_latlon.nc')

# %%
ds_era = xr.open_dataset('/work/bm1183/m301049/icon_hcap_data/publication/vertical_vel/era5_vel_2.grb')

# %% calculate pressure velocity
omega = - ds_icon['wa'] * 9.81 * ds_icon['rho'] 

# %% plot hist of w_500
fig, ax = plt.subplots(figsize=(5, 2.5))
bins=np.arange(-1.2, 1.2, 0.1)
hist_era = np.histogram(ds_era['w'].values.flatten(), bins=bins, density=True)[0]
hist_icon = np.histogram(omega.values.flatten(), bins=bins, density=True)[0]
ax.stairs(hist_era, bins, label='ERA5', color='k', linewidth=3)
ax.stairs(hist_icon, bins, label='ICON Control', color=definitions[2]['jed0011'], linewidth=1.5,)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel(r'$\omega_{\mathrm{500}}$ / Pa s$^{-1}$')
ax.set_ylabel('P($\omega_{\mathrm{500}}$)')
ax.legend(frameon=False)
fig.savefig('plots/vertical_vel_hist.pdf', bbox_inches='tight')

# %%
skew_era = np.var(ds_era['w'].values.flatten())
skew_icon = np.var(omega.values.flatten())
print(f"ERA5: skew={skew_era:.2f}")
print(f"ICON: skew={skew_icon:.2f}")


# %%
