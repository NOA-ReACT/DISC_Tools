#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical plotting module for EarthCARE-Ground validation analysis.

Provides visualization functions for profile statistics and scatter plot analysis.

Main functions: plot_profile_statistics, create_profile_figure, create_scatter_plots
Dependencies: matplotlib, scipy, stat_calc, statconfig, statio

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
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from scipy import stats
import sklearn
from sklearn.linear_model import LinearRegression

# Local EarthCARE tools
sys.path.append('/home/akaripis/earthcare')
from ectools_noa import ecio, ecplot as ecplt, colormaps as clm
from valtools.valconfig import DEFAULT_CONFIG_L1, DEFAULT_CONFIG_L2
from valtools.valio import *
from valtools.valplot import *

# Local validation tools
from statconfig import DEFAULT_CONFIG_ST
from stat_calc import get_variable_properties, calculate_scatter_stats
from statio import filter_dataset_by_values, remove_outliers

# =============================================================================
# PROFILE PLOTTING FUNCTIONS
# =============================================================================

def plot_profile_statistics(ds, ax, variable, plot_type='mean', hmax=None, 
                           remove_outliers=True, outlier_threshold=3.0, 
                           color='blue', label=None):
    """
    Create a profile statistics vertical plot.
    
    Parameters
    ----------
    ds : xarray.Dataset | Dataset containing statistics variables
    ax : matplotlib.Axes | Axes object to plot on
    variable : str | Variable name to plot
    plot_type : str | Type of plot ('mean' or 'std'), default 'mean'
    hmax : float | Maximum height for y-axis, default None
    remove_outliers : bool | Whether to remove outliers, default True
    outlier_threshold : float | Z-score threshold for outlier removal, default 3.0
    color : str | Plot color, default 'blue'
    label : str | Plot label, default None
        
    Returns
    -------
    bool | True if plot created successfully, False otherwise
    """
    # Determine variable and error names based on plot_type
    if plot_type == 'mean':
        varname = variable + '_mean'
        error = variable + '_std'
        xlabel = 'Mean'
    else:
        varname = variable + '_std'
        error = variable + '_std'
        xlabel = 'Standard Deviation'
    
    # Check if variables exist in the dataset
    if varname not in ds or error not in ds:
        print(f"Warning: Variables '{varname}' or '{error}' not found in the dataset. Skipping plot.")
        return False
        
    # Get data arrays
    var_data = ds[varname]
    err_data = ds[error]
    
    sc_factor, units, plot_title, xlim = get_variable_properties(variable, plot_type='profile')
    
    # Apply scaling factor
    var_data = var_data * sc_factor
    err_data = err_data * sc_factor
    
    if DEFAULT_CONFIG_ST['GROUPING_METHOD'] == 'km_bins':
        # Extract height values from km_bin coordinate
        height_values = var_data['km_bin'].values
        
        # Convert bin labels to numeric heights (midpoints)
        y_heights = []
        for bin_label in height_values:
            if '-' in bin_label:
                lower, upper = map(int, bin_label.split('-'))
                y_heights.append((lower + upper) / 2)
            else:
                # Handle non-standard bin labels
                print(f"Warning: Non-standard bin label: {bin_label}")
                nums = re.findall(r'\d+', bin_label)
                if nums:
                    y_heights.append(float(nums[0]))
                else:
                    y_heights.append(float('nan'))
        # Convert to numpy array
        y_heights = np.array(y_heights)
    else:
        # For height grouping, convert from meters to kilometers
        y_heights = var_data['height'].values / 1000.0
    
    # Get the actual data values
    var_values = var_data.values
    err_values = err_data.values
    
    # Remove outliers if requested
    if remove_outliers and len(var_values) > 3:
        # Calculate z-scores
        mean = np.mean(var_values)
        std = np.std(var_values)
        if std > 0:  # Avoid division by zero
            z_scores = np.abs((var_values - mean) / std)
            
            # Identify non-outliers
            non_outliers = z_scores < outlier_threshold
            
            # Filter data
            if np.sum(non_outliers) > 0:  # Make sure we have some data left
                print(f"Removing {np.sum(~non_outliers)} outliers out of {len(var_values)} points")
                var_values = var_values[non_outliers]
                err_values = err_values[non_outliers]
                y_heights = y_heights[non_outliers]
            else:
                print("Warning: Outlier removal would eliminate all data points. Keeping original data.")

    # Create the plot
    # Colorblind-safe blue (Okabe-Ito palette)
    cb_blue = '#0077BB'
    gray_band = '#a6cee3'
    
    # Plotting section
    if plot_type == 'mean':
        # Plot with error bars
        ax.errorbar(var_values, y_heights, xerr=err_values, fmt='o', capsize=5,
                    markersize=6, color=cb_blue, ecolor=gray_band, label=label)
    else:
        # Plot without error bars
        ax.errorbar(var_values, y_heights, fmt='o', capsize=5,
                    markersize=6, color=cb_blue, ecolor='gray_band', alpha=0.5, label=label)
    
    # Set labels and title only for the first plot (to avoid overwriting)
    if ax.get_xlabel() == '':
        ax.set_xlabel(f'{units}', fontsize=14)
        ax.set_ylabel('Height (km)', fontsize=14)
        ax.set_title(plot_title, fontsize=16)
    
    if hmax is None:
        hmax = 20  # default to 20 km
    
    ax.set_ylim(0, hmax)
    
    # Set y-ticks every 1 km
    ytick_positions = np.arange(0, hmax + 0.1, 1)
    ax.set_yticks(ytick_positions)
    # Set x-axis limits if provided
    ax.set_xlim(xlim)
    
    # Add grid and reference line
    ax.grid(True, linestyle=':', alpha=0.9)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
    return True


def create_profile_figure(plotting_stats_unfiltered, plotting_stats_filtered=None, 
                         variables=None, plot_type='mean', 
                         title="Statistics Plot", save_path=None, figsize=(22, 10),
                         remove_outliers=False, outlier_threshold=5.0):
    """
    Create figure for vertical statistics profiles comparison.
    
    Parameters
    ----------
    plotting_stats_unfiltered : xarray.Dataset | Statistics dataset for plotting
    plotting_stats_filtered : xarray.Dataset | Optional filtered statistics dataset, default None
    variables : list | List of variables to plot, default None uses config
    plot_type : str | Type of plot ('mean' or 'std'), default 'mean'
    title : str | Figure title, default "Statistics Plot"
    save_path : str | Path to save figure, default None
    figsize : tuple | Figure size, default (22, 10)
    remove_outliers : bool | Whether to remove outliers, default False
    outlier_threshold : float | Outlier threshold, default 5.0
        
    Returns
    -------
    matplotlib.figure.Figure | The created figure
    """
    if variables is None:
        variables = DEFAULT_CONFIG_ST['VARIABLES']
        
    # Number of axes based on the number of variables
    axes_number = len(variables)
    
    # Use appropriate height limits based on grouping method
    if DEFAULT_CONFIG_ST['GROUPING_METHOD'] == 'km_bins':
        hmax_values = DEFAULT_CONFIG_ST['HMAX_KM_BINS']
    else:
        hmax_values = DEFAULT_CONFIG_ST['HMAX_HEIGHT']
        
    # Ensure hmax_values has the right length
    if len(hmax_values) < axes_number:
        hmax_values = hmax_values + [hmax_values[-1]] * (axes_number - len(hmax_values))
    
    # Create the visualization
    fig = plt.figure(figsize=figsize)
    
    # Create width ratios list (all 1s)
    width_ratios = [1] * axes_number
    
    # Create GridSpec
    gs = GridSpec(1, axes_number, figure=fig, width_ratios=width_ratios, wspace=0.4, top=0.85)
    
    # Create list to store axes
    axes = []
    
    # Create subplot axes
    for i in range(axes_number):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        print(f'Processing variable {i+1}/{axes_number}: {variables[i]}')
        
        # Plot unfiltered data
        plot_profile_statistics(plotting_stats_unfiltered, ax=ax, variable=variables[i], 
                               plot_type=plot_type, hmax=hmax_values[i],
                               remove_outliers=remove_outliers, 
                               outlier_threshold=outlier_threshold,
                               color='blue', label='All data')
        
        # Plot filtered data if available
        if plotting_stats_filtered is not None and DEFAULT_CONFIG_ST['PLOT_BOTH']:
            plot_profile_statistics(plotting_stats_filtered, ax=ax, variable=variables[i], 
                                   plot_type=plot_type, hmax=hmax_values[i],
                                   remove_outliers=remove_outliers, 
                                   outlier_threshold=outlier_threshold,
                                   color='red', label='Filtered data')
            ax.legend()

    print('Profile figure created successfully')

    # Add a main title to the figure
    plot_type_title = plot_type.capitalize()
    fig.suptitle(f"{plot_type_title} bias - {title}", 
             fontsize=20, 
             y=0.98)
    fig.text(0.5, 0.01, "Mean bias calculation for EC profiles from the station.\n"
                        "Scatter points = mean bias, error bars = std.", 
             ha='center', va='bottom', fontsize=12, 
             transform=fig.transFigure)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# =============================================================================
# SCATTER PLOTTING FUNCTIONS
# =============================================================================

def create_scatter_plots(gnd_data_list, sat_data_list, event_names, 
                        single_event_idx=None, apply_filtering=False,
                        variables=None, figsize=(16, 12), show_stats=True, 
                        save_path=None, use_axis_limits=True):
    """
    Create scatter plots from data lists with 1:1 and regression lines.
    
    Parameters
    ----------
    gnd_data_list : list | List of ground datasets
    sat_data_list : list | List of satellite datasets  
    event_names : list | List of event names
    single_event_idx : int | Index for single event plot, None for all events, default None
    apply_filtering : bool | Whether to apply filtering, default False
    variables : list | Variables to plot, default None uses config
    figsize : tuple | Figure size, default (16, 12)
    show_stats : bool | Whether to show statistics on plots, default True
    save_path : str | Path to save figure, default None
    use_axis_limits : bool | Whether to use predefined axis limits, default True
    
    Returns
    -------
    matplotlib.figure.Figure | The created scatter plot figure
    """
    if variables is None:
        variables = DEFAULT_CONFIG_ST['VARIABLES']

    # Get data and statistics from stat_calc module
    all_gnd_data, all_sat_data, statistics = calculate_scatter_stats(
        gnd_data_list, sat_data_list, event_names, 
        apply_filtering=apply_filtering, variables=variables)

    # Determine if single event or multi-event
    if single_event_idx is not None:
        title_suffix = f" - {event_names[single_event_idx]}"
        print(f"Creating single event scatter plot for: {event_names[single_event_idx]}")
    else:
        title_suffix = f" - All Events ({len(event_names)} events)"
        print("Creating multi-event scatter plot...")
    
    # Subplot layout
    n_vars = len(variables)
    if n_vars <= 4:
        nrows, ncols = 2, 2
    elif n_vars <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for i, variable in enumerate(variables):
        ax = axes[i]
        
        if not all_gnd_data[variable] or not all_sat_data[variable]:
            ax.text(0.5, 0.5, f'{variable}\nno valid data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{variable} - No Valid Data')
            continue
        
        gnd_vals = np.array(all_gnd_data[variable])
        sat_vals = np.array(all_sat_data[variable])
        
        scale_factor, units, var_label, axis_limits = get_variable_properties(variable, plot_type='scatter')
        gnd_vals_scaled = gnd_vals * scale_factor
        sat_vals_scaled = sat_vals * scale_factor
        
        if use_axis_limits and axis_limits is not None:
            xlim, ylim = axis_limits, axis_limits
        else:
            min_val = min(np.min(gnd_vals_scaled), np.min(sat_vals_scaled))
            max_val = max(np.max(gnd_vals_scaled), np.max(sat_vals_scaled))
            margin = (max_val - min_val) * 0.05
            xlim = ylim = (min_val - margin, max_val + margin)
        
        # Scatter plot
        ax.scatter(gnd_vals_scaled, sat_vals_scaled, alpha=0.6, s=15)
        
        # 1:1 line
        ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], 'r--', alpha=0.8, label='1:1 line')
        
        # Regression line
        if statistics[variable] and len(gnd_vals_scaled) > 1:
            slope = statistics[variable]['slope']
            intercept = statistics[variable]['intercept']
            x_line = np.linspace(xlim[0], xlim[1], 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'g-', linewidth=1.2, label='Fit')
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Statistics box
        if show_stats and statistics[variable] and len(gnd_vals_scaled) > 1:
            stats_dict = statistics[variable]
            stats_text = (f'R={stats_dict["correlation"]:.3f}\n'
                         f'RMSE={stats_dict["rmse"]:.3f}\n'
                         f'Bias={stats_dict["bias"]:.3f}\n'
                         f'Slope={stats_dict["slope"]:.2f}, Int={stats_dict["intercept"]:.2f}\n'
                         f'n={stats_dict["n_points"]}')
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Ground {var_label} {units}')
        ax.set_ylabel(f'Satellite {var_label} {units}')
        ax.set_title(var_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    filter_text = " (Filtered)" if apply_filtering else ""
    title = f"Ground vs Satellite Comparison{filter_text}{title_suffix}"
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
    
    return fig