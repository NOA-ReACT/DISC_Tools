#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data plotting module for EarthCARE analysis tools.


"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import numpy as np
from scipy.signal import savgol_filter

import pdb
from val_L2_dictionaries import*

# Set the Seaborn style and font
sns.set_theme(context='paper', style='white', palette='deep', font_scale=1.7, 
        color_codes=True)

def adjust_subplot_position(ax, x_offset=0.02, y_offset=0, width_scale=1, 
                            height_scale=0.98, twinx=False, twiny=False):
    """
    Adjust the position of a subplot and its twin axes (both twinx and twiny) 
    if they exist.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes   | The axis to adjust
    x_offset : float            | Offset to add to x position (default: 0.02)
    y_offset : float            | Offset to add to y position (default: 0)
    width_scale : float         | Scale factor for width (default: 1)
    height_scale : float        | Scale factor for height (default: 0.98)
    
    Returns
    -------
    None
    """
    pos = ax.get_position()
    new_position = [
        pos.x0 + x_offset,
        pos.y0 + y_offset,
        pos.width * width_scale,
        pos.height * height_scale
    ]
    
    # Find the parent figure
    fig = ax.get_figure()
    
    # Adjust main axis
    ax.set_position(new_position) 

    # Find existing twin axes
    if twinx or twiny:
        for other_ax in ax.figure.axes:
            if twinx and other_ax.bbox.bounds == ax.bbox.bounds and other_ax != ax:
                other_ax.set_position(new_position)
            if twiny and other_ax.bbox.bounds == ax.bbox.bounds and other_ax != ax:
                other_ax.set_position(new_position)


def plot_EC_profiles(ds, varname, hmax=15e3, ax=None, profile='EC', 
                    heightvar='sample_altitude', title=None,
                    lin_scale=True, log_scale=True, xlim=None, xlim_log=None,
                    yticks=True, xlabel=False, legend=False):
    """
   Plot EarthCARE profiles with error ranges.
   
   Parameters
   ----------
   ds : xarray.Dataset  | Dataset containing the variables to plot
   varname : str        | Name of the variable to plot  
   hmax : float         | Maximum height for y-axis in meters (default: 15000m)
   ax : matplotlib.axes | Axes to plot on. If None, creates new figure
   profile : str        | Type of profile ('EC' or 'SM') determining colors and labels
   timevar : str        | Name of time dimension (default: 'time')
   heightvar : str      | Name of height dimension (default: 'sample_altitude')
   title : str          | Plot title
   title_prefix : str   | Prefix for the title
   lin_scale : bool     | Whether to plot linear scale (default: True)
   log_scale : bool     | Whether to plot logarithmic scale (default: True)
   xlim : tuple         | (min, max) for linear x-axis
   xlim_log : tuple     | (min, max) for logarithmic x-axis
   yticks : bool        | Whether to show y-axis ticks (default: True)
   
   Returns
   -------
   fig : matplotlib.figure.Figure | The created figure (only if ax was None)
   ax : matplotlib.axes.Axes      | The axes used for plotting
    -------
  
    """
    # Define color schemes
    colors = {
        'EC': {
            'line': '#1f77b4',
            'error': '#99CCFF',
            'log': '#1f77b4',
            'labels': ('EC lin', 'EC log')
        },
        'SM': {
            'line': '#B68AC9',
            'error': '#B68AC9',
            'log': '#B68AC9',
            'labels': ('SM lin', 'SM log')
        }
    }
    
    if profile not in colors:
        raise ValueError(f'Profile must be one of {list(colors.keys())}')
    
    color_scheme = colors[profile]
    
    var = ds[varname].mean('along_track') * 1e6
    error =ds[f"{varname}_total_error"].mean('along_track')*1e6

    if profile == 'EC':
        or_height = ds[heightvar]-ds['geoid_offset']
        height = or_height.mean(dim='along_track')
        # height = ds[heightvar].mean(dim='along_track')
    else: 
        height = ds[heightvar].mean(dim='along_track')[::-1]
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 20))
    else:
    # Get the current figure if ax is provided
        fig = ax.figure
        
    lines_linear, labels_line = [], []
    lines_log, labels_log = [], []
    
    if lin_scale:
        main_line = ax.step(var, height,
                           color=color_scheme['line'],
                           label=color_scheme['labels'][0],
                           linewidth=2)
        
        ax.fill_betweenx(height,
                         var - error, var + error,
                         color=color_scheme['error'],
                         label=f"{color_scheme['labels'][0]} error",
                         alpha=0.2)
                         
        if xlim is not None:
            ax.set_xlim(xlim)
            
        ax.axvline(x=0, color='grey', linestyle='-', alpha=0.5, linewidth=1)           
        ax.tick_params(axis='both', which='major', labelsize=11, length=7)
        ax.tick_params(axis='both', which='minor', labelsize=10, length=4)
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.grid(True, which='minor', linestyle=':', alpha=0.1)
        
        if log_scale:
            ax_log = ax.twiny()
            
        lines_linear, labels_line = ax.get_legend_handles_labels()
    else:
        ax_log = ax
        
    if log_scale:
        log_line = ax_log.step(var, height,
                              color=color_scheme['log'],
                              linestyle='-',
                              linewidth=1.5,
                              label=color_scheme['labels'][1])
                              
        ax_log.fill_betweenx(height,
                            var - error, var + error,
                            color=color_scheme['error'],
                            label=f"{color_scheme['labels'][1]} error",
                            alpha=0.1)
                            
        ax_log.set_xscale('log')
        
        if xlim_log is not None:
            ax_log.set_xlim(xlim_log)
        
        #ax_log.axvspan(-0.005, 0.005, color='grey', linestyle='-', alpha=0.4,linewidth=1)          
        ax_log.tick_params(axis='both', which='major', labelsize=11, length=7)
        ax_log.tick_params(axis='both', which='minor', labelsize=10, length=4)
        ax_log.grid(True, which='major', linestyle='-', alpha=0.2)
        ax_log.grid(True, which='minor', linestyle=':', alpha=0.1)
        
        lines_log, labels_log = ax_log.get_legend_handles_labels()
    else:
        lines_log, labels_log = [], []
        
    ax.set_ylim([0, hmax])
    ytick_positions = np.linspace(0, hmax, int(hmax/2e3 + 1))
    ax.set_yticks(ytick_positions)
    
    if yticks:
        ax.set_yticklabels([f'{int(pos/1000)}' for pos in ytick_positions],
                          fontsize=12)
        ax.set_ylabel('Height a.s.l (km)', fontsize=16, labelpad=10)
    else:
        ax.set_yticklabels([])
        
    if xlabel:
        ax.set_xlabel('Att.Backscatter Signal (Mm$^{-1}$sr$^{-1}$)',
                     fontsize=16)
                     
    lines = lines_linear + lines_log
    labels = labels_line + labels_log
    
    if legend:
        ax.legend(lines, labels,
                 bbox_to_anchor=(1.01, 1),
                 loc='upper left',
                 borderaxespad=0,
                 fontsize=10)
                 
    if title:
        ax.set_title(f'{title}', fontsize=14, fontweight='bold')

def plot_AEBD_profiles(ds, varname, hmax=16e3, idx=None, ax=None, resolution=None,
                      profile='EC',  heightvar='height', title=None, lin_scale=True,
                      log_scale=False, xlim=None, xlim_log=None, yticks=True, 
                      xlabel=True, legend=False,  smoothing=False):
    
    """
   Plot EarthCARE profiles with error ranges. Dictionaries set up for: backscatter 
   coefficient, extinction coefficient, lidar ratio and volume depolarization. Other 
   parameters must be manually added to the dictionaries.
   
   Parameters
   ----------
   ds : xarray.Dataset    | Dataset containing the variables to plot
   varname : str          | Name of the variable to plot  
   hmax : float           | Maximum height for y-axis in meters (default: 15000m)
   ax : matplotlib.axes   | Axes to plot on. If None, creates new figure
   profile : str          | Type of profile ('EC' or 'SM') determining colors and labels
   heightvar : str        | Name of height dimension (default: 'sample_altitude')
   title : str            | Plot title
   title_prefix : str     | Prefix for the title
   lin_scale : bool       | Whether to plot linear scale (default: True)
   log_scale : bool       | Whether to plot logarithmic scale (default: True)
   xlim : tuple           | (min, max) for linear x-axis
   xlim_log : tuple       | (min, max) for logarithmic x-axis
   yticks : bool          | Whether to show y-axis ticks (default: True)
   
   Returns
   -------
   fig : matplotlib.figure.Figure | The created figure (only if ax was None)
   ax : matplotlib.axes.Axes      | The axes used for plotting
    -------
     """
    # Define color schemes
    colors = {
        'EC': {
            'line': '#1f77b4',
            'error': '#99CCFF',
            'log': '#0066CC',
            'labels': ('EC lin', 'EC log'),
            'lin_linewidth': '2.5',
            'log_linewidth': '2.5',
            'alpha': 0.4

        },
        'GND': {
            'line': '#B68AC9',
            'error': '#B68AC9',
            'log': '#B68AC9',
            'labels': ('GND lin', 'GND log'),
            'lin_linewidth': '2',
            'log_linewidth': '2',
            'alpha': 0.2
        }
    }
    labels = {
        'particle_backscatter_coefficient_355nm': r'$ \beta_{355}$_' + profile,
        'particle_extinction_coefficient_355nm': r'$ \alpha_{355}$_' + profile,
        'lidar_ratio_355nm': r'$ lr_{355}$_' + profile,
        'particle_linear_depol_ratio_355nm': r'$ \delta_{p355}$_' + profile
    }
    
    xlabels = {
        'particle_backscatter_coefficient_355nm': ' $\mathregular{[Μm^{-1} sr^{-1}]}$',
        'particle_extinction_coefficient_355nm': ' $\mathregular{[Μm^{-1} ]}$',
        'lidar_ratio_355nm': '[sr]',
        'particle_linear_depol_ratio_355nm': '[-] '
    }
      
    varname = f'{varname}_{resolution}_resolution' if resolution in ['medium', 'low'] else varname
    
    if idx is None:
        idx = slice(None)
        
    if profile not in colors:
        raise ValueError(f'Profile must be one of {list(colors.keys())}')
        
    color_scheme = colors[profile]

    if varname in ['particle_backscatter_coefficient_355nm',
                   'particle_extinction_coefficient_355nm',
                   'particle_backscatter_coefficient_355nm_low_resolution',
                   'particle_extinction_coefficient_355nm_low_resolution',
                   'particle_backscatter_coefficient_355nm_medium_resolution',
                   'particle_extinction_coefficient_355nm_medium_resolution']:
        var = ds[varname][idx] * 1e6
        error = ds[f'{varname}_error'][idx] * 1e6
        if smoothing:
            var = savgol_filter(var, window_length=31, polyorder=3)
    else:
        var = ds[varname][idx]
        if smoothing:
            var = savgol_filter(var, window_length=31, polyorder=3)
        error = ds[f'{varname}_error'][idx]

    if profile == 'EC':
        height = ds[heightvar][idx]
    else:
        height = ds[heightvar][::]
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    else:
    # Get the current figure if ax is provided
        fig = ax.figure
        
    lines_linear, labels_line = [], []
    lines_log, labels_log = [], []
    
    # First, remove any resolution suffix
    base_var = varname.split('_medium_resolution')[0].split('_low_resolution')[0]
    # Then remove the '_r' or '_k' suffix if present
    if base_var.endswith('_r') or base_var.endswith('_k'):
        base_var = base_var[:-2]  # Remove the last 2 characters ('_r' or '_k')
    
    # Use this cleaned variable name to get the label
    label = labels[base_var]
    
    if lin_scale:
        main_line = ax.step(var, height, color=color_scheme['line'],
                            label = labels[base_var],where='mid',
                            linewidth=color_scheme['lin_linewidth'])
                           
        ax.fill_betweenx(height, var - error, var + error, color=color_scheme['error'],
                          alpha= color_scheme['alpha'], step='mid')

        if xlim is not None:
            ax.set_xlim(xlim)
        ax.axvspan(-0.005, 0.005, color='grey', linestyle='--', linewidth=1)          
        ax.tick_params(axis='both', which='major', labelsize=11, length=10)
        ax.tick_params(axis='both', which='minor', labelsize=10, length=10)
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.grid(True, which='minor', linestyle=':', alpha=0.1)
        
        if log_scale:
            ax_log = ax.twiny()
            
        lines_linear, labels_line = ax.get_legend_handles_labels()
    else:
        ax_log = ax
        lines_linear, labels_line = [], []
        
    if log_scale:
        log_line = ax_log.step(var, height, color=color_scheme['log'], linestyle='-',
                              linewidth=color_scheme['log_linewidth'], 
                              label=color_scheme['labels'][1])
                              
        ax_log.fill_betweenx(height,
                            var - error, var + error, color=color_scheme['error'],
                            alpha= color_scheme['alpha'])
                            
        ax_log.set_xscale('log')
        
        if xlim_log is not None:
            ax_log.set_xlim(xlim_log)
            
        ax_log.axvspan(-0.005, 0.005, color='grey', linestyle='--', linewidth=1)          
        ax_log.tick_params(axis='x', which='major', labelsize=11, length=6)
        ax_log.tick_params(axis='x', which='minor', labelsize=10, length=4)
        ax_log.grid(True, which='major', linestyle='-', alpha=0.2)
        ax_log.grid(True, which='minor', linestyle=':', alpha=0.1)
        
        lines_log, labels_log = ax_log.get_legend_handles_labels()
    else:
        lines_log, labels_log = [], []
        
    ax.set_ylim([0, hmax])
    ytick_positions = np.linspace(0, hmax, int(hmax/2e3 + 1))
    ax.set_yticks(ytick_positions)
    
    if yticks:
        ax.set_yticklabels([f'{int(pos/1000)}' for pos in ytick_positions],
                          fontsize=12)
        ax.set_ylabel('Height a.s.l (km)', fontsize=16, labelpad=10)
    else:
        ax.set_yticklabels([])
        
    if xlabel:
        ax.set_xlabel(xlabels[varname.split('_medium_resolution')[0].split('_low_resolution')[0]],
                     fontsize=14)
        
    # Legend handling
    lines=lines_linear+lines_log
    labels=labels_line+labels_log
    ax.legend(lines, labels, loc='upper right', fontsize=14)

    if title:
        ax.set_title(f"{title}",fontsize=14, fontweight='bold')
        

def plot_AEBD_scatter(ds, varname, hmax=16e3, idx=None, ax=None, resolution=None,
                    title=None, yticks=True, plot_type='classification',
                    x_offset=None, legend_number=1, xlim=False):
    """
    Plots scatter profile for A-EBD product
    
    Parameters
    ----------
    ds:    xarray.Dataset        | EarthCARE AEBD data 
    cla_variable : str           | Name of the classification variable
    qs_variable : str            | Name of the quality status variable
    title : str                  | Title of the plot
    idx : int                    | Index for data selection
    ax :                         | Pre-defined ax for plotting. If None, creates
                                    new figure with subplots (optional)
    Returns
    -------
    fig : matplotlib.figure.Figure | The created figure (only if axes was None)
    axes : list                    | List of axes used for plotting
    """
    # Create figure and axes if not provided  
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    
    if not varname == 'quality_status':
        varname = f'{varname}_{resolution}_resolution' if resolution in ['medium', 'low'] else varname

    var = ds[varname][idx]
    height = ds['height'][idx]
    
    if plot_type == 'classification':
        color_dict = classification_color_dict
        label_dict = classification_dict
        legend_title = 'Classification'
        marker = 'o'
    else:
        color_dict = quality_status_color_dict
        label_dict = quality_status_dict
        legend_title = 'Quality Status'
        marker= 'v'
    
    if x_offset:
        ax.scatter(np.ones_like(var) * x_offset, height, marker=marker,
                  c=[color_dict[x] for x in var.values], s=10)
    else:
        ax.scatter(var, height, marker=marker, c=[color_dict[x] for x in var.values], s=15)
    
    unique_values = np.ma.unique(var)
    labels = [label_dict[x] for x in unique_values]
    colors = [color_dict[x] for x in unique_values]
    
    ax.set_ylim([0, hmax])
    if xlim:
        ax.set_xlim([-10,x_offset*1.5])
    ytick_positions = np.linspace(0, hmax, int(hmax/2e3 + 1))
    ax.set_yticks(ytick_positions)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x-axis ticks
    ax.tick_params(axis='y', which='both', right=False, labelright=False)  # Remove right y-axis ticks
    
    # No x labels since both are plotted dimensionless
    ax.set_xticklabels([])
    
    if yticks:
        ax.set_yticklabels([f'{int(pos/1000)}' for pos in ytick_positions],
                          fontsize=12)
        ax.set_ylabel('Height (km)', fontsize=16, labelpad=10)
    else:
        ax.set_yticklabels([])
    
    fake_handles = [mpatches.Patch(color=c) for c in colors]
    
    ax.legend(fake_handles, labels,
             loc='center left',
             bbox_to_anchor=(1.02, 0.8 if legend_number == 1 else 0.5),
             framealpha=0.5,
             prop={'size': 8},
             title=legend_title,
             frameon=True,
             title_fontproperties={'weight': 'bold', 'size': 9})
             
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
        
def plot_AEBD_cla_qs(ds, cla_variable, qs_variable, hmax, title, idx, resolution,
                     ax=None, yticks=False):
    """
    Plots classification profile together with quality status
    
    Parameters
    ----------
    ds:    xarray.Dataset        | EarthCARE AEBD data 
    cla_variable : str           | Name of the classification variable
    qs_variable : str            | Name of the quality status variable
    title : str                  | Title of the plot
    idx : int                    | Index for data selection
    ax :                         | Pre-defined ax for plotting. If None, creates 
                                    new figure with subplots (optional)
    Returns
    -------
    fig : matplotlib.figure.Figure | The created figure (only if axes was None)
    axes : list                    | List of axes used for plotting
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 15))
        ax2 = ax.twinx()
    else:
        ax2 = ax.twinx()
        
    plot_AEBD_scatter(ds=ds, varname=cla_variable, hmax=hmax, ax=ax,resolution= resolution,
                     title=title, idx=idx, plot_type='classification',
                     x_offset=None, legend_number=1, yticks=yticks)
                     
    plot_AEBD_scatter(ds=ds, varname=qs_variable, hmax=hmax, ax=ax2,resolution= resolution,
                     title=None, idx=idx, plot_type='quality',
                     x_offset=45, legend_number=2, yticks=yticks, xlim=True)
       
def plot_paired_profiles(ec_data, sim_data, variable, hmax, ax=None, 
                        heightvar='sample_altitude', title=None, xlim=None, 
                        xlim_log=None, yticks=True,xlabel=False,legend=False,
                        lin_scale=True, log_scale=True):
    """
    Plot paired EC and simulation profiles on the same axes.
    
    Parameters
    ----------
    ec_data : xarray.Dataset    | Dataset containing EC profiles
    sim_data : xarray.Dataset   | Dataset containing simulation profiles or second EC dataset
    variable : str              | Name of the variable to plot
    hmax : float               | Maximum height for y-axis
    ax : matplotlib.axes.Axes   | Axes to plot on. If None, creates new figure and axes (optional)
    heightvar : str            | Name of height variable (default: 'sample_altitude')
    title : str                | Plot title (optional)
    xlim : tuple               | X-axis limits for linear scale (optional)
    xlim_log : tuple           | X-axis limits for log scale (optional)
    yticks : bool              | Whether to show y-axis ticks (default: False)
    xlabel : bool              | Whether to show x-axis label (default: False)
    legend : bool              | Whether to show legend (default: False)
    lin_scale : bool           | Whether to use linear scale (default: True)
    log_scale : bool           | Whether to use log scale (default: True)
    
    Returns
    -------
    fig : matplotlib.figure.Figure | The created figure (only if ax was None)
    ax : matplotlib.axes.Axes      | The axes used for plotting
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 25))
        
    # Plot EC profiles
    plot_EC_profiles(ec_data, variable, hmax, ax, profile='EC',
                    heightvar=heightvar,
                    title=variable.split('_')[0].capitalize() if title else None,
                    lin_scale=lin_scale, log_scale=log_scale,
                    xlim=xlim, xlim_log=xlim_log,
                    yticks=yticks, xlabel=xlabel, legend=legend)
                    
    plot_EC_profiles(sim_data, variable, hmax, ax, profile='SM',
                    heightvar=heightvar, title=None, lin_scale=lin_scale, 
                    log_scale=log_scale, xlim=xlim, xlim_log=xlim_log,
                    yticks=yticks, xlabel=xlabel, legend=legend)

def plot_profile_comparison(ec_data, sim_data, variables, axes=None, hmax=15e3,
                            heightvar='sample_altitude', title=True,
                            xlim=None, xlim_log=(1e-1, 1e1), lin_scale=True,
                            log_scale=True):
    
    """
   Plots parallel profile comparisons for all variables provided.
   
   Parameters
   ----------
   ec_data : xarray.Dataset    | EarthCARE data 
   sim_data : xarray.Dataset   | Simulation data
   variables : list            | List of variable names to plot
   axes : list                 | Pre-defined axes for plotting. If None, creates 
                                 new figure with subplots (optional)
   hmax : float               | Maximum height for plots in meters (default: 15000m)
   heightvar : str            | Name of height dimension (default: 'sample_altitude') 
   title : bool               | Whether to show titles (default: True)
   xlim : list of tuples      | List of x-axis limits for linear scale (optional)
   xlim_log : tuple or list   | List of x-axis limits for log scale. Single tuple 
                               or list of tuples (optional)
   lin_scale : bool           | Whether to use linear scale (default: True)
   log_scale : bool           | Whether to use log scale (default: True)

   Returns
   -------
   fig : matplotlib.figure.Figure | The created figure (only if axes was None)
   axes : list                    | List of axes used for plotting
    """
    
    # Create figure and axes if not provided
    fig = None
    if axes is None:
        n_vars = len(variables)
        fig, axes = plt.subplots(nrows=1,ncols=n_vars, figsize=(8, 15), 
                                gridspec_kw={'hspace': 0.3})
        # Ensure axes is always a list
        if n_vars == 1:
            axes = [axes]
    elif len(variables) != len(axes):
        raise ValueError(f"Number of variables ({len(variables)}) must match number of axes ({len(axes)})")

    # Handle xlim_log if it's a single tuple
    if isinstance(xlim_log, tuple):
        xlim_log = [xlim_log] * len(variables)
        

    for i, (variable, ax) in enumerate(zip(variables, axes)):
        current_xlim = xlim[i] if xlim is not None else None
        current_xlim_log = xlim_log[i] if xlim_log is not None else (1e-1, 1e1)
       
        plot_paired_profiles(ec_data=ec_data, sim_data=sim_data, variable=variable,
                             hmax=hmax, ax=ax,  heightvar=heightvar, title=title,
                             xlim=current_xlim, xlim_log=current_xlim_log,
                             yticks=(i == 0), xlabel=(i == 1), legend=(i==1),
                             lin_scale=lin_scale, log_scale=log_scale)
    
    if fig is not None:
        fig.tight_layout()
    
def plot_orbit_map(latitudes, longitudes, station_name, station_coordinates, 
                    shortest_distance, max_distance, buffer=1.4, lat2=None, 
                    lon2=None, distance_idx_nearest=None, idx=None, ax=None):
    """
    Plots the satellite orbit path on a map with a 'max_distance' radius circle around 
    the station.
    
    Parameters:
    - latitudes: List or array of satellite latitudes.
    - longitudes: List or array of satellite longitudes.
    - station_name: Name of the ground station to plot.
    - station_coordinates: Tuple (latitude, longitude) of the ground station.
    - buffer: Degrees to extend the map boundaries around the satellite path.
    - ax: Matplotlib axis to use for the plot. If None, creates a new figure.
    """
    
    # Center the map around the station coordinates
    station_lat, station_lon = station_coordinates
    
    lat_min = station_lat - 1.5 * buffer
    lat_max = station_lat + 1.5 * buffer
    lon_min = station_lon - buffer
    lon_max = station_lon + 1.2*buffer
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 20),
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                 crs=ccrs.PlateCarree())
                 
    ax.plot(longitudes, latitudes, color='black', lw=2,
            label='EC Orbit', transform=ccrs.PlateCarree())
            
    if distance_idx_nearest is not None:
        ax.plot(longitudes[distance_idx_nearest],
                latitudes[distance_idx_nearest],
                color='orangered', lw=2, label='Overpass',
                transform=ccrs.PlateCarree())
                
    def create_circle_coords(center_lat, center_lon, radius_km):
        R = 6371
        d = radius_km
        angular_radius = d / R
        n_points = 100
        angles = np.linspace(0, 2*np.pi, n_points)
        circle_lats = []
        circle_lons = []
        
        for angle in angles:
            lat1 = np.radians(center_lat)
            lon1 = np.radians(center_lon)
            
            lat2 = np.arcsin(np.sin(lat1) * np.cos(angular_radius) +
                           np.cos(lat1) * np.sin(angular_radius) * np.cos(angle))
            lon2 = lon1 + np.arctan2(np.sin(angle) * np.sin(angular_radius) * np.cos(lat1),
                                   np.cos(angular_radius) - np.sin(lat1) * np.sin(lat2))
                                   
            circle_lats.append(np.degrees(lat2))
            circle_lons.append(np.degrees(lon2))
            
        return circle_lats, circle_lons
        
    circle_lats, circle_lons = create_circle_coords(station_lat, station_lon, max_distance)
    
    poly = plt.Polygon(np.column_stack((circle_lons, circle_lats)),
                      facecolor='red', alpha=0.2,
                      transform=ccrs.PlateCarree(),
                      label=f'{max_distance} radius')
    ax.add_patch(poly)
    
    ax.scatter(station_coordinates[1], station_coordinates[0],
              color='red', s=200, label=station_name,
              marker='o', transform=ccrs.PlateCarree())
              
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.LAND, facecolor='#f5f2e8')  # Light beige/cream color
    ax.add_feature(cfeature.OCEAN, facecolor='#a8d5e2')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.2,
                     color='black', alpha=0.5, linestyle='--',
                     )
    
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    
    from geopy.distance import geodesic
   
    if idx == slice(None) or idx == None:
        ax.set_title(f'Min. distance: \n {shortest_distance:.1f} km',
                    fontsize=15, pad=8)
    else:
        
        profile_lat = lat2[idx]
        profile_lon = lon2[idx]
        
        # Calculate distance using geopy's geodesic distance
        station_point = (station_lat, station_lon)
        profile_point = (profile_lat, profile_lon)
        
        profile_distance = geodesic(station_point, profile_point).kilometers
        
        ax.set_title(f'Min. distance: {shortest_distance:.1f} km\nProfile distance: {profile_distance:.1f} km',
                    fontsize=15, pad=8)

        
    ax.legend(loc='lower left', fontsize=9)
    
    plt.show()
