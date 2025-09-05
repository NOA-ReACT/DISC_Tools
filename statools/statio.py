#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Input/Output module for statistical analysis.

This module handles all data loading, preprocessing, and preparation operations
for the statistical analysis of the EarthCARE to ground data. It provides functions to:

- Load and process ground-based lidar data (PollyXT network)
- Load and crop EarthCARE satellite data (AEBD products)
- Regrid datasets to common height coordinates
- Apply data filtering and smoothing
- Handle variable renaming and standardization

The module serves as the data pipeline foundation, ensuring consistent data
preparation across all analysis workflows. All functions return xarray datasets
with standardized coordinate systems and variable names.

Key Functions:
    load_all_events: Load multiple events for statistical analysis
    data_preparation_single_event: Process individual event data
    regrid_ds1_to_ds2: Align datasets to common height grid
    filter_dataset_by_values: Apply cloud/quality filtering
    smooth_ground_profiles: Apply Savitzky-Golay smoothing

Dependencies:
    - ectools_noa: ecio.py for EC data loading
    - valio: data preparation functions
    - statconfig: Configuration parameters

Notes:
    - All height coordinates are in meters
    - Ground data is averaged in time dimension
    - Satellite data can be spatially averaged or profile-selected
    - Missing data handling follows CF conventions (NaN values)

Created on: Sun Jul 20 08:46:44 2025

@author: Andreas Karipis
@affiliation: National Observatory of Athens (NOA), ReACT
@contact: akaripis@noa.gr
@version: 1.0
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
from scipy.signal import savgol_filter

# Local EarthCARE tools
sys.path.append('/home/akaripis/earthcare')
from ectools_noa import ecio, ecplot as ecplt, colormaps as clm
from valtools.valconfig import DEFAULT_CONFIG_L1, DEFAULT_CONFIG_L2
from valtools.valio import *
from valtools.valplot import *

# Local validation tools  
from statconfig import DEFAULT_CONFIG_ST

rename_vars = {
    'aerBsc_raman_355': 'particle_backscatter_coefficient_355nm',
    'uncertainty_aerBsc_raman_355': 'particle_backscatter_coefficient_355nm_error',
    'aerExt_raman_355': 'particle_extinction_coefficient_355nm',
    'uncertainty_aerExt_raman_355': 'particle_extinction_coefficient_355nm_error',
    'aerLR_raman_355': 'lidar_ratio_355nm',
    'uncertainty_aerLR_raman_355': 'lidar_ratio_355nm_error',
    'parDepol_raman_355': 'particle_linear_depol_ratio_355nm',
    'uncertainty_parDepol_raman_355': 'particle_linear_depol_ratio_355nm_error',
    'aerBsc_klett_355': 'particle_backscatter_coefficient_355nm',
    'uncertainty_aerBsc_klett_355': 'particle_backscatter_coefficient_355nm_error',
    'parDepol_klett_355': 'particle_linear_depol_ratio_355nm',
    'uncertainty_parDepol_klett_355': 'particle_linear_depol_ratio_355nm_error',
}
# A. READ ALL EVENTS AND RETURN DATA LISTS

def load_all_events(event_list, network, max_distance, klett=True, raman=True,
                    certain_profiles=False,height_min=None, height_max=None ):
    """
    Load all events and return lists of ground and satellite data.
    
    Parameters
    ----------
    event_list : list | List of event directory paths containing EarthCARE and ground data
    network : str | Ground station network identifier
    max_distance : float | Maximum spatial distance between satellite and ground station (km)
    klett : bool | Whether to use Klett retrieval method, default True
    raman : bool | Whether to use Raman retrieval method, default True
    certain_profiles : bool | If True, use only closest satellite profiles, default False
    height_min : float | Minimum height for analysis (meters), default None
    height_max : float | Maximum height for analysis (meters), default None
        
    Returns
    -------
    tuple | (gnd_data_list, sat_data_list, event_names) where all are lists
    """
    
    gnd_data_list = []
    sat_data_list = []
    event_names = []

    for event_path in event_list:
        try:
            event_name = event_path.split('/')[-1]
            print(f"Loading event: {event_name}")
            
            paths = build_paths(event_path, network, 'L2')
            
            # Load raw/unfiltered data
            regridded_gnd_prof, aebd_reg = data_preparation_single_event(
                paths['AEBD'], paths['GND'], network, max_distance, klett, raman,certain_profiles,
                height_min, height_max )
                    
            gnd_data_list.append(regridded_gnd_prof)
            sat_data_list.append(aebd_reg)
            event_names.append(event_name)
            
            print(f"Successfully loaded event: {event_name}")
            
        except Exception as e:
            print(f"Error loading event {event_path}: {e}")
            continue
    
    print(f"Total events loaded: {len(event_names)}")
    return gnd_data_list, sat_data_list, event_names

def data_preparation_single_event(datapath, gndpath, network, max_distance, klett=True, raman=True, 
                                      certain_profiles=False, height_min=None, height_max=None):
    """
    Load and prepare ground and satellite data for a single event.
    
    Parameters
    ----------
    datapath : str | Path to EarthCARE AEBD data file
    gndpath : str | Path to ground-based lidar data file
    network : str | Ground station network identifier
    max_distance : float | Maximum distance for satellite data selection (km)
    klett : bool | Use Klett retrieval method, default True
    raman : bool | Use Raman retrieval method, default True
    certain_profiles : bool | Use only closest satellite profiles, default False
    height_min : float | Minimum height for analysis (meters), default None
    height_max : float | Maximum height for analysis (meters), default None
        
    Returns
    -------
    tuple | (regridded_gnd_prof, aebd_reg) where both are xarray.Dataset
    """
    
    # Load ground data
    gnd_quicklook, gnd_profile, station_name, \
        station_coordinates = load_ground_data(network, gndpath, 'L2',
                                               scc_term='b0355')
    #pdb.set_trace()
    # Load and crop EarthCARE product
    aebd, aebd_50km, shortest_time, aebd_baseline, distance_idx_nearest, \
        dst_min, s_dist_idx, aebd_100km = load_crop_EC_product(
            datapath, station_coordinates, product='AEBD',
            max_distance=max_distance, second_trim=True, second_distance=110)
 
    polly_raman, polly_klett = read_pollynet_profile(gnd_profile, data=True)

   # pdb.set_trace()
    
    if certain_profiles == 'True' or certain_profiles == True:  # Handle both cases
        aebd_50km = aebd_50km.isel(along_track = slice(s_dist_idx-4,s_dist_idx+5))
            
    if raman is False: 
        polly_raman = None
        ds = polly_klett
    elif klett is False:
        polly_klett = None
        ds = polly_raman
    else:
        # Rename variables in ground profile
        ds = gnd_profile
        
    ds = ds.isel(method=0, reference_height=0)
    ds = ds.mean(dim='time')
    
    if DEFAULT_CONFIG_ST['SMOOTH_GROUND_DATA']:
        ds = smooth_ground_profiles(ds)
        #aebd_50km = smooth_ground_profiles(aebd_50km)
    
    # Only rename variables that exist in both ds and rename_vars
    variables_to_rename = {k: v for k, v in rename_vars.items() if k in ds}
    ds = ds.rename(variables_to_rename)
    
    aebd_50km_inverted = aebd_50km.isel(JSG_height=slice(None, None, -1))
    
    regridded_gnd_prof, aebd_reg = regrid_ds1_to_ds2(ds, aebd_50km_inverted, drop_other_dim=True)
    
    # Keep only the dimensions you want
    dims_to_keep = ['height', 'along_track']
    dims_to_collapse = [dim for dim in aebd_reg.dims if dim not in dims_to_keep]
    if dims_to_collapse:
        select_dict = {dim: 0 for dim in dims_to_collapse}
        aebd_reg = aebd_reg.isel(select_dict)
    
        
    # Apply height trimming if specified
    if height_min is not None or height_max is not None:
        # Create height mask
        height_mask = True
        if height_min is not None:
            height_mask = height_mask & (regridded_gnd_prof.height >= height_min)
        if height_max is not None:
            height_mask = height_mask & (regridded_gnd_prof.height <= height_max)
        
        # Apply trimming to both datasets
        regridded_gnd_prof = regridded_gnd_prof.where(height_mask, drop=True)
        aebd_reg = aebd_reg.where(height_mask, drop=True)
    

    return regridded_gnd_prof, aebd_reg


def regrid_ds1_to_ds2(ds1, ds2, ds1_heightvar='height', ds1_heightdim='height',
                      ds2_heightvar='height', ds2_heightdim='JSG_height',
                      output_heightdim='height',drop_other_dim=True):
    """
    Regrid first dataset to match the height grid of the second dataset.
    
    Parameters
    ----------
    ds1 : xarray.Dataset | Dataset to be regridded
    ds2 : xarray.Dataset | Dataset providing target height grid
    ds1_heightvar : str | Height variable name in ds1, default 'height'
    ds1_heightdim : str | Height dimension name in ds1, default 'height'
    ds2_heightvar : str | Height variable name in ds2, default 'height'
    ds2_heightdim : str | Height dimension name in ds2, default 'JSG_height'
    output_heightdim : str | Output height dimension name, default 'height'
    drop_other_dim : bool | Whether to drop non-matching dimensions, default True
        
    Returns
    -------
    tuple | (ds1_regridded, ds2_trimmed) where both are xarray.Dataset
    """
    
    # Determine the height range of ds1
    min_height = ds1[ds1_heightvar].min().item()
    max_height = ds1[ds1_heightvar].max().item()
    
    # Find best EC height indices that correspond to both min and max heights of GND
    best_jsg_max_idx = None
    best_jsg_min_idx = None
    min_distance_max = float('inf')
    min_distance_min = float('inf')
      
    # Iterate through all height indices in ds2
    for jsg_idx in range(ds2.dims[ds2_heightdim]):
        # Get mean height for this height index
        mean_height = ds2[ds2_heightvar].isel({ds2_heightdim: jsg_idx}).mean().values
        
        # Calculate distance to max target height
        distance_max = abs(mean_height - max_height)
        # Update if this is closer than previous best match for max
        if distance_max < min_distance_max:
            min_distance_max = distance_max
            best_jsg_max_idx = jsg_idx
        
        # Calculate distance to min target height
        distance_min = abs(mean_height - min_height)
        # Update if this is closer than previous best match for min
        if distance_min < min_distance_min:
            min_distance_min = distance_min
            best_jsg_min_idx = jsg_idx

    idx_target = (best_jsg_max_idx)-best_jsg_min_idx
    if idx_target >200:
        idx_target =idx_target-1
        best_jsg_max_idx-=1

    # Trim dataset 2 to the range we want
    ds2_trimmed = ds2.isel({ds2_heightdim: slice(best_jsg_min_idx, best_jsg_max_idx)})
    # Perform the interpolation
    target_heights = np.linspace(min_height, max_height, idx_target)

    ds1_regridded = ds1.interp({ds1_heightdim: target_heights})
    #pdb.set_trace()
    if output_heightdim:
        ds1_regridded = ds1_regridded.rename({ds1_heightdim: output_heightdim})
        ds2_trimmed = ds2_trimmed.rename({ds2_heightdim: output_heightdim})
    return ds1_regridded, ds2_trimmed

def filter_dataset_by_values(ds, filter_var, filter_values):
    """
    Filter dataset by setting specified values in a variable to NaN.
    
    Parameters
    ----------
    ds : xarray.Dataset | Input dataset to filter
    filter_var : str | Name of variable containing filter criteria
    filter_values : list | Values to filter out (set to NaN)
        
    Returns
    -------
    xarray.Dataset | Filtered dataset with specified values masked as NaN
    """
    if not isinstance(filter_values, (list, tuple, set)):
        filter_values = [filter_values]

    # Create mask for filtering
    mask = xr.concat([ds[filter_var] == val for val in filter_values], dim='filter_vals').any(dim='filter_vals')

    # Define variables to exclude from masking
    exclude_vars = {'height', 'time', 'lat', 'lon', 'latitude', 'longitude'}
    coords = set(ds.coords)

    # Only apply mask to non-coordinate, non-excluded variables
    data_vars = {}
    for var in ds.data_vars:
        if var not in exclude_vars and var not in coords:
            data_vars[var] = ds[var].where(~mask)
        else:
            data_vars[var] = ds[var]

    return xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)

def smooth_ground_profiles(ds):    
    """
    Apply Savitzky-Golay smoothing to ground-based profile data.
    
    Parameters
    ----------
    ds : xarray.Dataset | Input dataset containing profile variables with height dimension
        
    Returns
    -------
    xarray.Dataset | Smoothed dataset with same structure as input
    """
    from scipy.signal import savgol_filter
    
    # Fixed parameters
    window_length = 71
    polyorder = 3
    
    # Make copy to avoid modifying original
    ds_smoothed = ds.copy()
    
    # Apply Savgol filter to each variable along height dimension
    for var_name in ds.data_vars:
        if 'height' in ds[var_name].dims:
            # Get the data
            var_data = ds[var_name].values
            
            # Check if we have enough data points
            if len(var_data) >= window_length:
                # Apply Savitzky-Golay filter
                smoothed_data = savgol_filter(var_data, 
                                             window_length=window_length, 
                                             polyorder=polyorder)
                # Update the dataset
                ds_smoothed[var_name].values = smoothed_data
            else:
                print(f"Warning: Not enough data points for {var_name}. Skipping smoothing.")
    
    return ds_smoothed

def remove_outliers(gnd_vals, sat_vals, method='iqr', factor=1.5):
    """
    Remove statistical outliers from paired ground and satellite measurements.
    
    Parameters
    ----------
    gnd_vals : array-like | Ground-based measurement values
    sat_vals : array-like | Satellite measurement values
    method : str | Outlier detection method: 'iqr' or 'zscore', default 'iqr'
    factor : float | Threshold factor for outlier definition, default 1.5
        
    Returns
    -------
    tuple | (gnd_vals_clean, sat_vals_clean) where both are numpy arrays with outliers removed
    """
    
    if method == 'iqr':
        # Calculate IQR for both datasets
        def get_iqr_mask(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return (data >= lower_bound) & (data <= upper_bound)
        
        # Apply IQR filter to both datasets
        gnd_mask = get_iqr_mask(gnd_vals)
        sat_mask = get_iqr_mask(sat_vals)
        
        # Keep only points where both values are within bounds
        combined_mask = gnd_mask & sat_mask
        
    elif method == 'zscore':
        from scipy import stats
        # Calculate z-scores
        gnd_z = np.abs(stats.zscore(gnd_vals))
        sat_z = np.abs(stats.zscore(sat_vals))
        
        # Keep points where both z-scores are below threshold
        combined_mask = (gnd_z < factor) & (sat_z < factor)
    
    return gnd_vals[combined_mask], sat_vals[combined_mask]