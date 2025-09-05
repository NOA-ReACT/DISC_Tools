#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical calculation module for EarthCARE-Ground validation analysis.

Provides functions for computing bias statistics between satellite and ground
measurements, with consistent data masking for profile and scatter analysis.

Main functions: calculate_bias, calculate_km_statistics, profile_statistics, calculate_scatter_stats
Dependencies: statconfig, statio, numpy, xarray, scipy

@author: Andreas Karipis - NOA ReaCT
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import sys
import re
import pdb

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy import stats

# Local EarthCARE tools
sys.path.append('/home/akaripis/earthcare')
from ectools_noa import ecio, ecplot as ecplt, colormaps as clm
from valtools.valconfig import DEFAULT_CONFIG_L1, DEFAULT_CONFIG_L2
from valtools.valio import *
from valtools.valplot import *

# Local validation tools
from statconfig import DEFAULT_CONFIG_ST
from statio import filter_dataset_by_values, remove_outliers

# =============================================================================
# BIAS CALCULATION FUNCTIONS
# =============================================================================

def calculate_bias(sat_ds, gnd_ds, variables=None, km_bins=True):
    """
    Calculate bias between satellite and ground measurements with consistent masking.
    
    Parameters
    ----------
    sat_ds : xarray.Dataset | Satellite dataset
    gnd_ds : xarray.Dataset | Ground dataset
    variables : list | List of variables to analyze, default None uses config
    km_bins : bool | Whether to create kilometer bins, default True
        
    Returns
    -------
    xarray.Dataset | Dataset containing bias values with optional km_bin coordinate
    """
    if variables is None:
        variables = DEFAULT_CONFIG_ST['VARIABLES']

    bias_ds = xr.Dataset()

    for varname in variables:
        if varname in sat_ds and varname in gnd_ds:
            sat_var = sat_ds[varname]
            gnd_var = gnd_ds[varname]

            # Broadcast to common shape
            sat_b, gnd_b = xr.broadcast(sat_var, gnd_var)

            # Build mask to match scatter filtering: finite and non-zero on both
            mask = (
                sat_b.notnull() & gnd_b.notnull() &
                (sat_b != 0) & (gnd_b != 0)
            )

            # Apply mask, compute difference only on valid pairs
            sat_m = sat_b.where(mask)
            gnd_m = gnd_b.where(mask)
            bias_var = sat_m - gnd_m

            bias_ds[varname] = bias_var

    # Optional km binning
    if km_bins:
        if "height" not in bias_ds.coords:
            raise ValueError("`height` coordinate not found in bias_ds; cannot km-bin.")
        heights_km = bias_ds["height"].values / 1000.0

        # Handle NaNs robustly
        hmax = np.nanmax(heights_km) if np.isfinite(heights_km).any() else 0.0
        km_edges = np.arange(0, np.ceil(hmax) + 1, 1)
        if km_edges.size < 2:  # Ensure at least one bin
            km_edges = np.array([0, max(1, int(np.ceil(hmax) + 1))])

        km_labels = [f"{int(km_edges[i])}-{int(km_edges[i+1])}" for i in range(len(km_edges)-1)]

        # Assign each height to a km bin
        km_bin_idx = np.digitize(heights_km, km_edges) - 1  # 0-based
        km_labels_assigned = [
            km_labels[idx] if 0 <= idx < len(km_labels) else "overflow"
            for idx in km_bin_idx
        ]

        bias_ds = bias_ds.assign_coords(km_bin=("height", km_labels_assigned))

    return bias_ds


def calculate_km_statistics(bias_ds_with_km, groupby='km_bins'):
    """
    Calculate statistics from bias data grouped by height bins.
    
    Parameters
    ----------
    bias_ds_with_km : xarray.Dataset | Bias dataset with height binning
    groupby : str | Grouping method: 'km_bins' or 'height', default 'km_bins'
        
    Returns
    -------
    xarray.Dataset | Dataset with statistical measures per height bin
    """
    # Store statistics for each kilometer bin
    km_stats = {}
    for var_name in bias_ds_with_km.data_vars:
        # Calculate statistics within each km bin
        if groupby == 'km_bins':
            grouped = bias_ds_with_km[var_name].groupby("km_bin")
            km_stats[var_name] = {
                'mean': grouped.mean(dim=["along_track", "height"]),
                'std': grouped.std(dim=["along_track", "height"]),
                'count': grouped.count(dim=["along_track", "height"]),
                'sem': grouped.std(dim=["along_track", "height"]) / np.sqrt(grouped.count(dim=["along_track", "height"])),
                'std_error': grouped.std(dim=["along_track", "height"]) / np.sqrt(2 * grouped.count(dim=["along_track", "height"]))
            }
        else:
            if groupby == 'height':
                grouped = bias_ds_with_km[var_name].groupby("height")
                km_stats[var_name] = {
                    'mean': grouped.mean(dim=["along_track"]),
                    'std': grouped.std(dim=["along_track"]),
                    'count': grouped.count(dim=["along_track"]),
                    'sem': grouped.std(dim=["along_track"]) / np.sqrt(grouped.count(dim=["along_track"])),
                    'std_error': grouped.std(dim=["along_track"]) / np.sqrt(2 * grouped.count(dim=["along_track"]))
                }
    
    # Create dataset with the kilometer bin statistics
    km_stats_ds = xr.Dataset()
    for var_name, stats in km_stats.items():
        km_stats_ds[f"{var_name}_mean"] = stats['mean']
        km_stats_ds[f"{var_name}_sem"] = stats['sem']
        km_stats_ds[f"{var_name}_std"] = stats['std']
        km_stats_ds[f"{var_name}_std_error"] = stats['std_error']
        km_stats_ds[f"{var_name}_count"] = stats['count']
    
    return km_stats_ds

# =============================================================================
# PROFILE STATISTICS FUNCTIONS
# =============================================================================

def profile_statistics(gnd_data_list, sat_data_list, event_names, apply_filtering=False):
    """
    Calculate profile statistics from multiple events.
    
    Parameters
    ----------
    gnd_data_list : list | List of ground datasets
    sat_data_list : list | List of satellite datasets  
    event_names : list | List of event names
    apply_filtering : bool | Whether to apply filtering, default False
        
    Returns
    -------
    xarray.Dataset | Combined statistics dataset ready for plotting
    """
    print(f"Calculating profile statistics (filtering={apply_filtering})...")
    
    km_stats_ds_list = []
    
    for i, (gnd_ds, sat_ds, event_name) in enumerate(zip(gnd_data_list, sat_data_list, event_names)):
        try:
            print(f"Processing event {i+1}/{len(event_names)}: {event_name}")
            
            # Apply filtering if requested
            if apply_filtering and DEFAULT_CONFIG_ST['ENABLE_FILTERING']:
                filter_var = DEFAULT_CONFIG_ST['FILTER_VARIABLE']
                filter_values = DEFAULT_CONFIG_ST['FILTER_VALUES']
                if filter_var in sat_ds:
                    sat_ds = filter_dataset_by_values(sat_ds, filter_var, filter_values)
            
            # Calculate bias
            km_bins = (DEFAULT_CONFIG_ST['GROUPING_METHOD'] == 'km_bins')
            bias_ds = calculate_bias(sat_ds, gnd_ds, 
                                   variables=DEFAULT_CONFIG_ST['VARIABLES'],
                                   km_bins=km_bins)
            
            # Calculate statistics
            km_stats_ds = calculate_km_statistics(bias_ds, groupby=DEFAULT_CONFIG_ST['GROUPING_METHOD'])
            
            # Add event coordinate
            km_stats_ds = km_stats_ds.assign_coords(event=event_name)
            km_stats_ds = km_stats_ds.expand_dims('event')
            
            km_stats_ds_list.append(km_stats_ds)
            
        except Exception as e:
            print(f"Error processing event {event_name}: {e}")
            continue
    
    if not km_stats_ds_list:
        raise ValueError("No events were successfully processed")
    
    # Combine all datasets and calculate mean statistics across events
    reference_height = km_stats_ds_list[0]['height'].values

    for i in range(len(km_stats_ds_list)):
        km_stats_ds_list[i] = km_stats_ds_list[i].assign_coords(height=reference_height)
    combined_km_stats_ds = xr.concat(km_stats_ds_list, dim='event')

    plotting_stats = combined_km_stats_ds.mean(dim='event')
    
    print(f"Profile statistics calculated for {len(km_stats_ds_list)} events")
    return plotting_stats

# =============================================================================
# SCATTER STATISTICS FUNCTIONS
# =============================================================================

def get_variable_properties(variable, plot_type='scatter'):
    """
    Get scaling factor, units, label, and axis limits for a given variable.
    
    Parameters
    ----------
    variable : str | Variable name to get properties for
    plot_type : str | Type of plot ('scatter' or 'profile'), default 'scatter'
        
    Returns
    -------
    tuple | (scale_factor, units, label, axis_limits) where axis_limits can be None
    """
    if plot_type == 'profile':
        if variable == 'particle_backscatter_coefficient_355nm':
            return 1e6, ' $\mathregular{[Mm^{-1} sr^{-1}]}$', 'Bsc.coef.', [-2, 5]
        elif variable == 'particle_extinction_coefficient_355nm':
            return 1e6, ' $\mathregular{[Mm^{-1}]}$', 'Ext.coef.', [-100, 140]
        elif variable == 'lidar_ratio_355nm':
            return 1, 'sr', 'Lidar ratio', [-200, 200]
        elif variable == 'particle_linear_depol_ratio_355nm':
            return 1, '-', 'Par.depol.ratio', [-1.5, 1.5]
        else:
            return 1, '[-]', variable, None
    
    else:  # plot_type == 'scatter'
        if variable == 'particle_backscatter_coefficient_355nm':
            return 1e6, '[Mmâ»Â¹srâ»Â¹]', 'Backscatter Coef.', (-1, 6.)
        elif variable == 'particle_extinction_coefficient_355nm':
            return 1e6, '[Mmâ»Â¹]', 'Extinction Coef.', (-20, 150)
        elif variable == 'lidar_ratio_355nm':
            return 1, '[sr]', 'Lidar Ratio', (-20, 200)
        elif variable == 'particle_linear_depol_ratio_355nm':
            return 1, '[-]', 'Depolarization Ratio', (-0.1, 0.6)
        else:
            return 1, '[-]', variable, None


def calculate_scatter_stats(gnd_data_list, sat_data_list, event_names, apply_filtering=False,
                            variables=None):
    """
    Prepare and calculate statistics for scatter plots.
    
    Parameters
    ----------
    gnd_data_list : list | List of ground datasets
    sat_data_list : list | List of satellite datasets  
    event_names : list | List of event names
    apply_filtering : bool | Whether to apply filtering, default False
    variables : list | Variables to extract, default None uses config
    
    Returns
    -------
    tuple | (all_gnd_data, all_sat_data, statistics) where statistics is a dict per variable
    """
    if variables is None:
        variables = DEFAULT_CONFIG_ST['VARIABLES']
    event_indices = range(len(event_names))

    all_gnd_data = {var: [] for var in variables}
    all_sat_data = {var: [] for var in variables}
    
    # Data collection phase
    for i in event_indices:
        try:
            gnd_ds = gnd_data_list[i]
            sat_ds = sat_data_list[i]
            
            # Apply filtering if requested
            if apply_filtering and DEFAULT_CONFIG_ST['ENABLE_FILTERING']:
                filter_var = DEFAULT_CONFIG_ST['FILTER_VARIABLE']
                filter_values = DEFAULT_CONFIG_ST['FILTER_VALUES']
                if filter_var in sat_ds:
                    sat_ds = filter_dataset_by_values(sat_ds, filter_var, filter_values)
            
            sat_ds = sat_ds.mean(dim='along_track')

            for variable in variables:
                if variable in gnd_ds and variable in sat_ds:
                    gnd_vals = gnd_ds[variable].values.flatten()
                    sat_vals = sat_ds[variable].values.flatten()
                    
                    # Remove invalid data
                    valid_mask = ~(np.isnan(gnd_vals) | np.isnan(sat_vals) | 
                                   (gnd_vals == 0) | (sat_vals == 0))
                    
                    if np.any(valid_mask):
                        gnd_vals_clean = gnd_vals[valid_mask]
                        sat_vals_clean = sat_vals[valid_mask]
                        
                        # Remove outliers if requested
                        if DEFAULT_CONFIG_ST['REMOVE_OUTLIERS'] and len(gnd_vals_clean) > 10:
                            gnd_vals_clean, sat_vals_clean = remove_outliers(
                                gnd_vals_clean, sat_vals_clean, 
                                method=DEFAULT_CONFIG_ST['OUTLIER_METHOD'],
                                factor=DEFAULT_CONFIG_ST['OUTLIER_FACTOR'])
                        
                        all_gnd_data[variable].extend(gnd_vals_clean)
                        all_sat_data[variable].extend(sat_vals_clean)
                        
        except Exception as e:
            print(f"Error processing event {event_names[i]}: {e}")
            continue
    
    # Statistics calculation phase
    statistics = {}
    for variable in variables:
        if all_gnd_data[variable] and all_sat_data[variable]:
            gnd_vals_all = np.array(all_gnd_data[variable])
            sat_vals_all = np.array(all_sat_data[variable])
            
            scale_factor, units, var_label, axis_limits = get_variable_properties(variable, plot_type='scatter')
            gnd_vals_scaled = gnd_vals_all * scale_factor
            sat_vals_scaled = sat_vals_all * scale_factor
            
            if len(gnd_vals_scaled) > 1:
                correlation, _ = stats.pearsonr(gnd_vals_scaled, sat_vals_scaled)
                rmse = np.sqrt(np.mean((sat_vals_scaled - gnd_vals_scaled)**2))
                bias = np.mean(sat_vals_scaled - gnd_vals_scaled)
                slope, intercept, _, _, _ = stats.linregress(gnd_vals_scaled, sat_vals_scaled)
                
                statistics[variable] = {
                    'correlation': correlation,
                    'rmse': rmse,
                    'bias': bias,
                    'slope': slope,
                    'intercept': intercept,
                    'n_points': len(gnd_vals_scaled)
                }
            else:
                statistics[variable] = None
        else:
            statistics[variable] = None
    
    return all_gnd_data, all_sat_data, statistics