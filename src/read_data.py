import xarray as xr

runs = ["jed0011", "jed0022", "jed0033"]
experiments = {
    "jed0011": "control",
    "jed0022": "plus4K",
    "jed0033": "plus2K",
    "jed0224": "const_ozone",
}


def get_path():
    """
    Get the path to the data directory.
    You will have to modify this function to point to the location where you have stored the data.
    """
    return "/work/bm1183/m301049/icon_hcap_data/publication"


def load_iwp_distributions():
    """
    Load the IWP distributions for the model.

    Returns
    -------
    dict
        Dictionary containing the IWP distributions.
    """
    path = get_path()
    iwp_dists = xr.open_dataset(f"{path}/distributions/iwp_distributions.nc")

    return iwp_dists


def load_iwp_distributions_30():
    """
    Load the IWP distributions for the model for 30 days.

    Returns
    -------
    dict
        Dictionary containing the IWP distributions.
    """
    path = get_path()
    iwp_dists = xr.open_dataset(f"{path}/distributions/iwp_dist_30_days.nc")

    return iwp_dists


def load_iwp_distributions_60():
    """
    Load the IWP distributions for the model for 60 days.

    Returns
    -------
    dict
        Dictionary containing the IWP distributions.
    """
    path = get_path()
    iwp_dists = xr.open_dataset(f"{path}/distributions/iwp_dist_60_days.nc")

    return iwp_dists


def load_daily_cycle_dists():
    """
    Load the distributions of deep convective clouds over local time and incoming SW.
    Returns
    -------
    dict
        Dictionary containing the daily cycle distributions.

    xarray.Dataarray
        Dataarray containing the incoming SW radiation.
    """
    path = get_path()
    daily_cycle_dists = {}
    for run in runs:
        daily_cycle_dists[run] = xr.open_dataset(
            f"{path}/distributions/{run}_deep_clouds_daily_cycle.nc"
        )

    SW_in = xr.open_dataarray(f"{path}/incoming_sw/SW_in_daily_cycle.nc")
    return daily_cycle_dists, SW_in

def load_hr_components():
    """
    Load the mean heating rate components.

    Returns
    -------
    dict
        Dictionary containing the mean flux divergence.
    
    dict
        Dictionary containing the mean heating rate.

    dict
        Dictionary containing the mean density.
    """
    path = get_path()
    mean_fluxdiv = {}
    mean_hr = {}
    mean_rho = {}
    for run in runs:
        mean_fluxdiv[run] = xr.open_dataarray(
            f"{path}/flux_conv/{run}_flux_divergence.nc"
        )
        mean_hr[run] = xr.open_dataarray(f"{path}/flux_conv/{run}_heating_rate.nc")
        mean_rho[run] = xr.open_dataarray(f"{path}/flux_conv/{run}_air_density.nc")

    return mean_fluxdiv, mean_hr, mean_rho


def load_vgrid():
    """
    Load the vertical grid for the model.

    Returns
    -------
    xarray.Dataset
        Dataset containing the vertical grid.
    """
    path = get_path()
    vgrid = (
        xr.open_dataset(
            f"{path}/grid/atm_vgrid_angel.nc"
        )
        .mean("ncells")
        .rename({"height": "height_2", "height_2": "height"})
    )

    return vgrid


def load_cre():
    """
    Load  HCRE.

    Returns
    -------
    dict
        Dictionary containing the CRE data.
    """
    path = get_path()
    cre_data = {}
    for run in runs:
        cre_data[run] = xr.open_dataset(
            f"{path}/cre/{run}_cre_raw.nc"
        )

    return cre_data

def load_hc_temp():
    """
    Load the temperature at the top of the high clouds.

    Returns
    -------
    dict
        Dictionary containing the temperature at the top of the high clouds.
    """
    path = get_path()
    hc_temp = xr.open_dataset(f"{path}/hc_temp/hc_temperatures.nc")

    return hc_temp

def load_hr_and_cf():
    """
    Load Heating rates and cloud fraction profiles binned by IWP.
    
    Returns
    -------
    dict
        Dictionary containing the net heating rates.
    dict
        Dictionary containing the longwave heating rates.
    dict
        Dictionary containing the shortwave heating rates.
    dict
        Dictionary containing cloud fraction"""
    
    path = get_path()
    hrs_binned = xr.open_dataset(f"{path}/heating_rates/hr_net.nc")
    hrs_sw_binned = xr.open_dataset(f"{path}/heating_rates/hr_sw.nc")
    hrs_lw_binned = xr.open_dataset(f"{path}/heating_rates/hr_lw.nc")
    cf_binned = xr.open_dataset(f"{path}/heating_rates/cf.nc")


    return hrs_binned, hrs_lw_binned, hrs_sw_binned, cf_binned

def load_sw_metrics():
    """
    Load shortwave metrics binned by IWP.
    
    Returns
    -------
    dict
        Dictionary containing the shortwave down at TOA.
    dict
        Dictionary containing the time difference to noon.
    dict
        Dictionary containing the latitude of deep convective clouds.
    """
    path = get_path()
    sw_down_binned = xr.open_dataset(f"{path}/incoming_sw/sw_incoming.nc")
    rad_time_binned = xr.open_dataset(f"{path}/incoming_sw/time_difference.nc")
    lat_binned = xr.open_dataset(f"{path}/incoming_sw/lat_distance.nc")

    return sw_down_binned, rad_time_binned, lat_binned

def load_lc_fractions():
    """
    Load low cloud fractions binned by IWP.
    
    Returns
    -------
    dict
        Dictionary containing the low cloud fractions with tuning factor.
    dict
        Dictionary containing the low cloud fractions without tuning factor.
    """
    path = get_path()
    lc_binned = {}
    lc_binned_raw = {}
    for run in runs:
        lc_binned[run] = xr.open_dataarray(f"{path}/lc_fraction/{run}_lc_frac.nc")
        lc_binned_raw[run] = xr.open_dataarray(f"{path}/lc_fraction/{run}_lc_frac_raw.nc")

    return lc_binned, lc_binned_raw

def load_stab_iris():
    """
    Load heating rate, stability, subsidence, and convergence profiles.
    Returns
    -------
    dict
        Dictionary containing the heating rate profiles.
    dict
        Dictionary containing the stability profiles.
    dict
        Dictionary containing the subsidence profiles.
    dict
        Dictionary containing the convergence profiles.
    """
    path = get_path()
    hrs = {}
    stab = {}
    subs = {}
    conv = {}
    for run in runs:
        hrs[run] = xr.open_dataset(f"{path}/stab_iris/{run}_heating_rate.nc")
        stab[run] = xr.open_dataarray(f"{path}/stab_iris/{run}_stability.nc")
        subs[run] = xr.open_dataarray(f"{path}/stab_iris/{run}_subsidence.nc")
        conv[run] = xr.open_dataarray(f"{path}/stab_iris/{run}_convergence.nc")
    
    subs_cont = xr.open_dataarray(f"{path}/stab_iris/jed0022_subsidence_const_hr.nc")
    conv_cont = xr.open_dataarray(f"{path}/stab_iris/jed0022_convergence_const_hr.nc")

    return hrs, stab, subs, conv, subs_cont, conv_cont

def load_t_profiles():
    """
    Load temperature profiles.
    Returns
    -------
    dict
        Dictionary containing the temperature profiles.
    """
    path = get_path()
    t_profiles = {}
    for run in [*runs, "jed2224"]:
        t_profiles[run] = xr.open_dataarray(f"{path}/t_profiles/{run}_t_profile_clearsky.nc")

    return t_profiles


def load_definitions():
    """
    Load the definitions for the runs, experiment names, colors, and labels.
    Returns
    -------
    tuple
        A tuple containing the runs, experiment names, colors, and labels.
    """

    runs = ["jed0011", "jed0033", "jed0022"]
    exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
    colors = {"jed0011": "#462d7b", "jed0022": "#c1df24", "jed0033": "#1f948a"}
    linestyles = {
        "jed0011": "-.",
        "jed0022": "-",
        "jed0033": "--",
    }
    sw_color = "#125fd3"
    lw_color = "#bd154a"
    net_color = "#000000"
    labels = {
        "jed0011": "Control",
        "jed0022": "+4 K",
        "jed0033": "+2 K",
    }
    return runs, exp_name, colors, labels, sw_color, lw_color, net_color, linestyles
