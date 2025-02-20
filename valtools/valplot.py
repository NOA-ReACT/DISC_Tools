#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 01:03:43 2025

Supporting plotting funtions for the 

@author: akaripis
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import numpy as np


from val_L2_dictionaries import*

# Set the Seaborn style and font
sns.set(context='paper', style='white', palette='deep', font_scale=1.7, 
        color_codes=True)

def adjust_subplot_position(ax, x_offset=0.02, y_offset=0, width_scale=1, 
                            height_scale=0.98, twinx=False, twiny=False):
    """
    Adjust the position of a subplot and its twin axes (both twinx and twiny) if they exist.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes    | The axis to adjust
    x_offset : float            | Offset to add to x position (default: 0.02)
    y_offset : float           | Offset to add to y position (default: 0)
    width_scale : float        | Scale factor for width (default: 1)
    height_scale : float       | Scale factor for height (default: 0.98)
    
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
            'line': '#0066CC',
            'error': '#99CCFF',
            'log': '#0066CC',
            'labels': ('EC lin', 'EC log')
        },
        'SM': {
            'line': '#009E73',
            'error': '#FFCC99',
            'log': '#FF8800',
            'labels': ('SM lin', 'SM log')
        }
    }
    
    if profile not in colors:
        raise ValueError(f'Profile must be one of {list(colors.keys())}')
    
    color_scheme = colors[profile]
    
    var = ds[varname].mean('along_track') * 1e6
    error =ds[f"{varname}_total_error"].mean('along_track')*1e6

    if profile == 'EC':
        height = ds[heightvar].mean(dim='along_track')
    else:
        height = ds[heightvar].mean(dim='along_track')[::-1]
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 20))
        
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
                         alpha=0.3)
                         
        if xlim is not None:
            ax.set_xlim(xlim)
            
        ax.tick_params(axis='both', which='major', labelsize=11, length=7)
        ax.tick_params(axis='both', which='minor', labelsize=10, length=4)
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.grid(True, which='minor', linestyle=':', alpha=0.1)
        
        if log_scale:
            ax_log = ax.twiny()
            
        lines_linear, labels_line = ax.get_legend_handles_labels()
    else:
        ax_log = ax
        lines_linear, labels_line = [], []
        
    if log_scale:
        log_line = ax_log.step(var, height,
                              color=color_scheme['log'],
                              linestyle='--',
                              linewidth=1.5,
                              label=color_scheme['labels'][1])
                              
        ax_log.fill_betweenx(height,
                            var - error, var + error,
                            color=color_scheme['error'],
                            label=f"{color_scheme['labels'][1]} error",
                            alpha=0.3)
                            
        ax_log.set_xscale('log')
        
        if xlim_log is not None:
            ax_log.set_xlim(xlim_log)
            
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
        ax.set_ylabel('Height (km)', fontsize=16, labelpad=10)
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
                      profile='EC', timevar='time', heightvar='height',
                      title=None, lin_scale=True, log_scale=False, xlim=None,
                      xlim_log=None, yticks=True, xlabel=True, legend=False):
    """
   Plot EarthCARE profiles with error ranges. Dictionaries set up for: backscatter coefficient, extinction coefficient, 
   lidar ratio and volume depolarization. Other parameters must be manually added to the dictionaries.
   
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
            'log_linewidth': '2',
            'alpha': 0.4

        },
        'GND': {
            'line': '#B68AC9',
            'error': '#B68AC9',
            'log': '#B68AC9',
            'labels': ('GND lin', 'GND log'),
            'lin_linewidth': '1.8',
            'log_linewidth': '1.3',
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
        'particle_extinction_coefficient_355nm': ' $\mathregular{[Μm^{-1} sr^{-1}]}$',
        'lidar_ratio_355nm': '[sr]',
        'particle_linear_depol_ratio_355nm': '[-] '
    }
    

    if resolution == 'high':
        suffix = ''
    elif resolution == 'medium':
        suffix = '_medium_resolution'
    else:
        suffix = '_low_resolution'
        
    varname = varname if resolution is None else varname + suffix
    
    if idx is None:
        idx = slice(None)
        
    if profile not in colors:
        raise ValueError(f'Profile must be one of {list(colors.keys())}')
        
    color_scheme = colors[profile]
    
    if varname in ['particle_backscatter_coefficient_355nm',
                   'particle_extinction_coefficient_355nm']:
        var = ds[varname][idx] * 1e6
        error = ds[f'{varname}_error'][idx] * 1e6
    else:
        var = ds[varname][idx]
        error = ds[f'{varname}_error'][idx]
        
    if profile == 'EC':
        height = ds[heightvar][idx]
    else:
        height = ds[heightvar][::]
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
        
    lines_linear, labels_line = [], []
    lines_log, labels_log = [], []
    
    if lin_scale:
        main_line = ax.step(var, height, color=color_scheme['line'],
                           label=labels[varname.split('_medium_resolution')[0].split('_low_resolution')[0]],
                           linewidth=color_scheme['lin_linewidth'])
                           
        ax.fill_betweenx(height, var - error, var + error, color=color_scheme['error'],
                         alpha= color_scheme['alpha'])
                         
        if xlim is not None:
            ax.set_xlim(xlim)
            
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
        log_line = ax_log.step(var, height, color=color_scheme['log'], linestyle='--',
                              linewidth=color_scheme['log_linewidth'], 
                              label=color_scheme['labels'][1])
                              
        ax_log.fill_betweenx(height,
                            var - error, var + error, color=color_scheme['error'],
                            alpha= color_scheme['alpha'])
                            
        ax_log.set_xscale('log')
        
        if xlim_log is not None:
            ax_log.set_xlim(xlim_log)
            
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
        ax.set_ylabel('Height (km)', fontsize=16, labelpad=10)
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
                    heightvar='height', title=None, yticks=True, plot_type='classification',
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
    ax :                         | Pre-defined ax for plotting. If None, creates new figure with subplots (optional)
    Returns
    -------
    fig : matplotlib.figure.Figure | The created figure (only if axes was None)
    axes : list                    | List of axes used for plotting
    """
    # Create figure and axes if not provided  
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    
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
        
def plot_AEBD_cla_qs(ds, cla_variable, qs_variable, hmax, title, idx, ax=None, yticks=False):
    """
    Plots classification profile together with quality status
    
    Parameters
    ----------
    ds:    xarray.Dataset        | EarthCARE AEBD data 
    cla_variable : str           | Name of the classification variable
    qs_variable : str            | Name of the quality status variable
    title : str                  | Title of the plot
    idx : int                    | Index for data selection
    ax :                         | Pre-defined ax for plotting. If None, creates new figure with subplots (optional)
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
        
    plot_AEBD_scatter(ds=ds, varname=cla_variable, hmax=hmax, ax=ax,
                     title=title, idx=idx, plot_type='classification',
                     x_offset=None, legend_number=1, yticks=yticks)
                     
    plot_AEBD_scatter(ds=ds, varname=qs_variable, hmax=hmax, ax=ax2,
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
   axes : list                 | Pre-defined axes for plotting. If None, creates new figure with subplots (optional)
   hmax : float               | Maximum height for plots in meters (default: 15000m)
   heightvar : str            | Name of height dimension (default: 'sample_altitude') 
   title : bool               | Whether to show titles (default: True)
   xlim : list of tuples      | List of x-axis limits for linear scale (optional)
   xlim_log : tuple or list   | List of x-axis limits for log scale. Single tuple or list of tuples (optional)
   lin_scale : bool           | Whether to use linear scale (default: True)
   log_scale : bool           | Whether to use log scale (default: True)

   Returns
   -------
   fig : matplotlib.figure.Figure | The created figure (only if axes was None)
   axes : list                    | List of axes used for plotting
    """
    
    # Create figure and axes if not provided
    if axes is None:
        n_vars = len(variables)
        fig, axes = plt.subplots(nrows=1,ncols=n_vars, figsize=(8, 15), 
                                gridspec_kw={'hspace': 0.3})
        # Ensure axes is always a list
        if n_vars == 1:
            axes = [axes]
    else:
        fig = None
        if len(variables) != len(axes):
            raise ValueError(f"Number of variables ({len(variables)}) must match number of axes ({len(axes)})")

    # Handle xlim_log if it's a single tuple
    if isinstance(xlim_log, tuple):
        xlim_log = [xlim_log] * len(variables)
        
    if len(variables) != len(axes):
        raise ValueError(f"Number of variables ({len(variables)}) must match number of axes ({len(axes)})")
    
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
                    shortest_distance, buffer=1.2, zoom_level=8, 
                    distance_idx_nearest=None, plot_station=True, ax=None):
    """
    Plots the satellite orbit path on a map with a 100km radius circle around the station.
    
    Parameters:
    - latitudes: List or array of satellite latitudes.
    - longitudes: List or array of satellite longitudes.
    - station_name: Name of the ground station to plot.
    - station_coordinates: Tuple (latitude, longitude) of the ground station.
    - buffer: Degrees to extend the map boundaries around the satellite path.
    - zoom_level: Zoom level for the map tiles.
    - plot_station: Whether to plot the ground station on the map.
    - ax: Matplotlib axis to use for the plot. If None, creates a new figure.
    """
    
    # Center the map around the station coordinates
    station_lat, station_lon = station_coordinates
    
    lat_min = station_lat - 1.5 * buffer
    lat_max = station_lat + 1.5 * buffer
    lon_min = station_lon - buffer
    lon_max = station_lon + buffer
    
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
        
    circle_lats, circle_lons = create_circle_coords(station_lat, station_lon, 100)
    
    poly = plt.Polygon(np.column_stack((circle_lons, circle_lats)),
                      facecolor='red', alpha=0.2,
                      transform=ccrs.PlateCarree(),
                      label='100km radius')
    ax.add_patch(poly)
    
    ax.scatter(station_coordinates[1], station_coordinates[0],
              color='red', s=200, label=station_name,
              marker='o', transform=ccrs.PlateCarree())
              
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
    
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
    
    ax.set_title(f'Min. distance: \n {shortest_distance:.1f} km',
                fontsize=15, pad=8)
    ax.legend(loc='lower left', fontsize=9)
    
    plt.show()

# def plot_EC_L1_comparison(anompath, simpath, sccfolderpath, pollyforlderpath, dstdir,
#                           network, lin_scale=True, log_scale=False, 
#                           max_distance=DEFAULT_CONFIG['MAX_DISTANCE'],
#                           hmax=DEFAULT_CONFIG['HMAX'], figsize=DEFAULT_CONFIG['FIGSIZE']):
    
#     """
#     This function loads data from multiple sources (EarthCARE ATLID, simulator, 
#     and ground station), processes it, and creates a multi-panel visualization 
#     comparing the different measurements.
    
#     Parameters
#     ----------
#     anompath : str                  |Path to the ANOM data file from EarthCARE
#     simpath : str                   |Path to the simulator data file
#     sccfolderpath : str             |Path to the folder containing ground station 
#                                     (SCC) data
#     pollyfolderpath: str            | Path to folder containing POLLYXT files
#     distdir : str                  |Directory where output figures will be saved
#     network: str                   | Gnd data network's data that are processed
#     max_distance : float, optional |Maximum distance in kilometers to consider for 
#                                       nearby points, by default 50
#     hmax : float, optional        |Maximum height for vertical axis in meters, 
#                                     default 16000
#     lin_scale : bool, optional    |Whether to use linear scale for profile plots, 
#                                     by default True
#     log_scale : bool, optional    |Whether to use logarithmic scale for profile 
#                                    plots, by default False
#     figsize : tuple, optional     |Figure size in inches (width, height), default 
#                                     (27, 15)
        
#     Returns
#     -------
#     matplotlib.figure.Figure
#         The generated comparison plot figure
        
#     Notes
#     -----
#     The function creates a complex figure with multiple panels:
#     - Left panels: Three quicklooks from ANOM data
#     - Center panels: Two quicklooks from ground station
#     - Right panels: Three profile comparisons and a map
    
#     The figure is automatically saved if distdir is provided.
#     """
#     # Load GND data
#     gnd_quicklook, station_name, station_coordinates = load_ground_data(network,
#                                                                         pollyforlderpath, 
#                                                                         sccfolderpath, 'L1')

#     # Load simulator data - no preproccessing needed
#     SIM = ecio.load_ANOM(simpath)
    
#     # Load and crop EC product
#     anom, anom_50km, shortest_time, baseline, distance_idx_nearest, dst_min, dist_idx, anom_100km = (
#         load_crop_EC_product( anompath, station_coordinates, 'ANOM', max_distance=max_distance,
#         second_trim=True, second_distance=100)
#         )
#     # Format overpass date
#     overpass_date = pd.Timestamp(shortest_time.item()).strftime('%d-%m-%Y %H:%M')
#     overpass_date = '2023-09-24 14:10:20' #mock value for dummy  files.
#     if network == 'POLLYXT':
#         gnd_quicklook = crop_polly_file(gnd_quicklook, overpass_date)
     
#     # Initialize the figure
#     fig = plt.figure(figsize=figsize)
#     gs = GridSpec( 9, 6,
#         figure=fig,
#         width_ratios=[1, 1, 0.005, 1.3, 1.3, 1.2],
#         height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1],
#         hspace=1.8,
#         wspace=0.55,
#         top=0.82
#     )
    
#     # Add main title
#     fig.suptitle(
#         f'EarthCARE A-NOM ({baseline}) Comparison with simulated data\n'
#         f' based on {station_name} measurements\n'
#         f'{overpass_date} UTC',
#         fontsize=24, weight='bold',  va='top', y=0.94)
    
#     # Create subplots
#     # Adjust the anom quicklook axis
#     ax1 = fig.add_subplot(gs[0:3, 0:2])
#     ax2 = fig.add_subplot(gs[3:6, 0:2])
#     ax3 = fig.add_subplot(gs[6:9, 0:2])
    
#     adjustments = {
#         ax1: {'width_scale': 1.05,'height_scale':0.95},
#         ax2: {'width_scale': 1.05},
#         ax3: {'width_scale': 1.05}
#     }

#     for ax, params in adjustments.items():
#         adjust_subplot_position(ax, **params)
        
#     # Plot anom quicklooks
#     ecplt.quicklook_ANOM(anom_100km, hmax=1.2 * hmax, dstdir=dstdir,
#                          axes=[ax1, ax2, ax3], comparison=True, 
#                          station=shortest_time )
    
#     # Adjust the scc quicklook axis
#     ax4 = fig.add_subplot(gs[0:4, 3:4])
#     ax5 = fig.add_subplot(gs[0:4, 4:5])

#     adjustments = {
#         ax4: {'x_offset': 0.003},
#         ax5: {'x_offset': 0.003}
#     }

#     for ax, params in adjustments.items():
#         adjust_subplot_position(ax, **params)
    
#     # Plot GND data
#     if network == 'EARLINET':
#         variables_q = ['range_corrected_signal', 'volume_linear_depolarization_ratio']
#         titles_q = [f'{station_name} range.cor.signal', f'{station_name} vol.depol.ratio']
#         plot_range = [[0, 15e8], [0, 0.2]]
#         heightvar = 'altitude'
#     elif network == 'POLLYXT':
#         variables_q = ['attenuated_backscatter_355nm', 'volume_depolarization_ratio_355nm']
#         titles_q = [f'{station_name} att.bsc', f'{station_name} vol.depol.ratio']
#         plot_range = [[0, 1.5e-5], [0, 0.3]]
#         heightvar= 'height'
#     else:
#         # Default case or error handling
#         raise ValueError(f"Unsupported network: {network}. Must be either 'EARLINET' or 'POLLYXT'")
        
#     axs = [ax4,ax5]
    
#     for i, (ax, variable, title, p_range) in enumerate(zip(axs, variables_q, titles_q, plot_range)):
#         ecplt.plot_gnd_2D(ax, gnd_quicklook, variable, ' ', heightvar=heightvar,
#                           cmap=clm.chiljet2, plot_scale='linear',
#                           plot_range=p_range, units='-', hmax=16e3,
#                           plot_position='bottom',
#                           title=title, comparison=True,
#                           scc=True, yticks=(i == 0), xticks=False)
#     # Adjust the profiles axis
#     ax6 = fig.add_subplot(gs[4:, 3])
#     ax7 = fig.add_subplot(gs[4:, 4])
#     ax8 = fig.add_subplot(gs[4:, 5])

#     adjustments = {
#         ax6: {'height_scale': 0.96, 'width_scale': 0.96},
#         ax7: {'height_scale': 0.96, 'width_scale': 0.96,'x_offset': -0.002},
#         ax8: {'height_scale': 0.96, 'width_scale': 0.96}
#     }

#     for ax, params in adjustments.items():
#         adjust_subplot_position(ax, **params)
        
#     # Define the profile axis ranges 
#     xlims = DEFAULT_CONFIG['DEFAULT_XLIMS'] if lin_scale else None
#     xlims_log = DEFAULT_CONFIG['DEFAULT_XLIMS_LOG'] if log_scale else None
    
#     # Define the variables that will be plotted. Must be 3. 
#     variables = [
#         'mie_attenuated_backscatter',
#         'rayleigh_attenuated_backscatter',
#         'crosspolar_attenuated_backscatter'
#     ]
    
#     # Plot variable profiles
#     plot_profile_comparison(anom_50km, SIM, variables, [ax6, ax7, ax8], 
#                             hmax=hmax,xlim=xlims, xlim_log=xlims_log, 
#                             lin_scale=lin_scale, log_scale=log_scale)

#     # Adjust map plot axis
#     ax9 = fig.add_subplot(gs[0:4, 5], projection=ccrs.PlateCarree())
#     adjust_subplot_position(
#         ax9,
#         x_offset=0.01,
#         y_offset=-0.04, 
#         height_scale=1.4,
#         width_scale=1.2
#     )

#     # Plot overpass map
#     plot_orbit_map(
#         anom['latitude'],
#         anom['longitude'],
#         station_name,
#         station_coordinates,
#         dst_min,
#         ax=ax9,
#         distance_idx_nearest=distance_idx_nearest
#     )
    
#     # Save figure if destination directory is provided
#     if dstdir:
#         srcfile_string = (
#             anom_50km.encoding['source'].split('/')[-1].split('.')[0]
#         )
#         dstfile = f'{overpass_date}_L1_intercomparison.png'
#         fig.savefig(f'{dstdir}/{dstfile}', bbox_inches='tight')
    
#     # Adjust layout
#     plt.tight_layout(rect=[0.1, 0.1, 0.88, 0.85])
#     fig.subplots_adjust(
#         top=0.82,
#         bottom=0.1,
#         left=0.1,
#         right=0.88
#     )
    
#     return fig


# def plot_sub_L2(idx, resolution, gnd_quicklooks, station_name, station_coordinates,
#                       aebd, aebd_50km, shortest_time, baseline,
#                       distance_idx_nearest, dst_min, s_dist_idx, aebd_100km, atc,
#                       atc_100km, gnd_profiles, dstdir, hmax,
#                       fig_scale , network, keyword='Raman', figsize=(35, 20)):
#     """
#     Creates L2 comparison plots between EarthCARE and ground data.
    
#     Parameters
#     ----------
#     idx: int                 | Index for profile selection
#     resolution: str          | Data resolution ('high', 'medium', 'low')
#     scc: xarray.Dataset      | SCC ground station data
#     station_name: str        | Name of ground station
#     station_coordinates: list| [latitude, longitude] of station
#     aebd: xarray.Dataset     | Full AEBD dataset
#     aebd_50km: xarray.Dataset| AEBD data within 50km of station
#     shortest_time: datetime  | Time of closest approach
#     baseline: str           | Processing baseline version
#     distance_idx_nearest: array| Indices of nearby points
#     dst_min: float          | Minimum distance to station
#     s_dist_idx: int         | Index of shortest distance point
#     aebd_100km: xarray.Dataset| AEBD data within 100km of station
#     atc: xarray.Dataset      | Full ATC dataset
#     atc_100km: xarray.Dataset| ATC data within 100km of station
#     gnd_profiles: xarray.Dataset | Ground-based profile data
#     hmax: float             | Maximum height for plots in meters (default: 16000)
#     network: str             | Ground network, for processing the data
#     keyword: str            | Type of ground data (default: 'Raman')
#     figsize: tuple          | Figure size in inches (default: (35, 20))
#     lin_scale: bool         | Use linear scale (default: True)
#     log_scale: bool         | Use logarithmic scale (default: False)
    
#     Returns
#     -------
#     fig: matplotlib.figure   | The generated comparison plot
#     """
#     if fig_scale == 'linear':
#         lin_scale = True
#         log_scale = False
#     elif fig_scale == 'log':
#         lin_scale = False
#         log_scale = True     
#     else:
#         lin_scale = True
#         log_scale = True     
        
#     time = (aebd_50km['time'])[idx]
#     overpass_time = pd.Timestamp(time.item()).strftime('%d-%m-%Y %H:%M:%S.%f')[:-4]
    
#     # Initialize figure
#     fig = plt.figure(figsize=figsize)
#     gs = GridSpec(10, 8, figure=fig, width_ratios=[1, 1, 1, 1.3, 1.3, 1.3, 1.3,
#                  1.2], height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], hspace=1.8,
#                  wspace=0.6, top=0.85)
    
#     # Add main title
#     fig.suptitle(f'EarthCARE A-EBD({baseline[0]}) & A-TC({baseline[1]}) Comparison with\n'
#                  f' {station_name} Ground Station  L2 {network} - {keyword} Retrieval \n'
#                  f'{overpass_time} UTC',
#                  fontsize=26, weight='bold', va='top', y=.96)


#     # Create and adjust quicklook axes
#     ax1 = fig.add_subplot(gs[0:2, 0:3])
#     ax2 = fig.add_subplot(gs[2:4, 0:3])
#     ax3 = fig.add_subplot(gs[4:6, 0:3])
#     ax4 = fig.add_subplot(gs[6:8, 0:3])
#     ax5 = fig.add_subplot(gs[8:10, 0:3])
    
#     adjustments = {
#         ax1: {'x_offset': -0.01, 'height_scale': 1},
#         ax2: {'x_offset': -0.01, 'height_scale': 1},
#         ax3: {'x_offset': -0.01, 'height_scale': 1},
#         ax4: {'x_offset': -0.01, 'height_scale': 1},
#         ax5: {'x_offset': -0.01, 'height_scale': 1.12}
#     }

#     for ax, params in adjustments.items():
#         adjust_subplot_position(ax, **params)
            
#     # Plot EBD and TC
#     ecplt.quicklook_AEBD(aebd_100km, resolution=resolution,  hmax=1.5*hmax if hmax < 30e3 else 30e3,
#                          dstdir=None, axes=[ax1, ax2, ax3, ax4, ax5],
#                          comparison=True, station=shortest_time)
    
#     ecplt.quicklook_ATC(atc_100km,  hmax=1.5*hmax if hmax < 30e3 else 30e3, resolution=resolution, dstdir=None,
#                         axes=ax5, comparison=True, station=shortest_time)
    
#     # Create and adjust SCC axes
#     ax6 = fig.add_subplot(gs[0:4, 3:5])
#     ax7 = fig.add_subplot(gs[0:4, 5:7])
    
#     adjustments = {
#         ax6: {'x_offset': 0.01, 'width_scale': 0.95,'height_scale': 0.95,'y_offset': 0.01,},
#         ax7: {'x_offset': 0, 'width_scale': 0.95,'height_scale': 0.95,'y_offset': 0.01,}
#     }

#     for ax, params in adjustments.items():
#         adjust_subplot_position(ax, **params)

#     # Plot GND data
#     if network == 'EARLINET':
#         variables_q = ['range_corrected_signal', 'volume_linear_depolarization_ratio']
#         titles_q = [f'{station_name} range.cor.signal', f'{station_name} vol.depol.ratio']
#         plot_range = [[0, 15e8], [0, 0.2]]
#         heightvar = 'altitude'
#         units=['-','-']
#     elif network == 'POLLYXT':
#         variables_q = ['quasi_bsc_532', 'quasi_pardepol_532']
#         titles_q = [f'{station_name} att.bsc', f'{station_name} par.depol.ratio']
#         plot_range = [[0, 15e-6], [0, 0.4]]
#         heightvar = 'height'
#         units = ['m⁻¹ sr⁻¹','-']
#     else:
#         # Default case or error handling
#         raise ValueError(f"Unsupported network: {network}. Must be either 'EARLINET' or 'POLLYXT'")
        
#     axs = [ax6,ax7]
    
#     for i, (ax, variable, title, p_range, unit) in enumerate(zip(axs, variables_q, titles_q, plot_range, units)):
#         ecplt.plot_gnd_2D(ax, gnd_quicklooks, variable, ' ', heightvar=heightvar,
#                           cmap=clm.chiljet2, plot_scale='linear',
#                           plot_range=p_range, units=unit,  hmax=hmax if hmax < 22e3 else 22e3,
#                           plot_position='bottom',
#                           title=title, comparison=True,
#                           scc=True, yticks=(i == 0), xticks=False)


#     # Create and adjust profile axes
#     ax8 = fig.add_subplot(gs[4:10, 3])
#     ax9 = fig.add_subplot(gs[4:10, 4])
#     ax10 = fig.add_subplot(gs[4:10, 5])
#     ax11 = fig.add_subplot(gs[4:10, 6:7])
#     ax12 = fig.add_subplot(gs[4:10, 7])
    
#     adjustments = {
#         ax8: {'x_offset': 0.05, 'height_scale': 1},
#         ax9: {'x_offset': 0.04, 'height_scale': 1},
#         ax10: {'x_offset': 0.02, 'height_scale': 1},
#         ax11: {'x_offset': 0.01, 'height_scale': 1},
#         ax12: {'height_scale': 1}
#     }

#     for ax, params in adjustments.items():
#         adjust_subplot_position(ax, **params)
    
#     # Define variables and axes for profiles
#     variables = [
#         'particle_backscatter_coefficient_355nm',
#         'particle_extinction_coefficient_355nm',
#         'lidar_ratio_355nm',
#         'particle_linear_depol_ratio_355nm'
#     ]
    
#     axes = [ax8, ax9, ax10, ax11]
    
#     # Define the profile axis ranges 
#     xlims = DEFAULT_CONFIG['DEFAULT_XLIMS'] if lin_scale else None
#     xlims_log = DEFAULT_CONFIG['DEFAULT_XLIMS_LOG'] if log_scale else None
    
#     # Plot AEBD profiles
#     titles = ['Bsc. Coef.', 'Ext. Coef.', 'Lidar Ratio', 'Lin. depol. ratio']
    
#     # Plot ground data if available
#     for i, (variable, ax) in enumerate(zip(variables, axes)):
#         if variable in gnd_profiles:
#             plot_AEBD_profiles(gnd_profiles, variable, ax=ax, lin_scale=lin_scale,
#                              log_scale=log_scale, profile='GND',
#                              yticks=(i == 0))  # Only True for first axis
            
#     for i, (variable, ax, title) in enumerate(zip(variables, axes, titles)):
#         plot_AEBD_profiles(aebd_50km, variable,hmax=hmax,resolution=resolution,
#                            ax=ax, lin_scale=lin_scale,idx=idx,
#                            log_scale=log_scale,title=title, profile='EC',
#                            xlim=xlims[i] if xlims else None,
#                            xlim_log=xlims_log[i] if xlims_log else None,
#                            yticks=(i == 0))  # Only True for first axis

#     # Plot classification and quality status
#     plot_AEBD_cla_qs(atc_100km, 'classification', 'quality_status',
#                      idx=idx, hmax=hmax, title='Classification & \nQuality Status',
#                      ax=ax12, yticks=False)
#     # Create and adjust map plot
#     ax_map = fig.add_subplot(gs[0:4, 7], projection=ccrs.PlateCarree())
#     adjust_subplot_position(ax_map, x_offset=0.01, y_offset=-0.08,
#                           height_scale=2.5, width_scale=1.4)

#     # Plot map
#     plot_orbit_map(aebd['latitude'], aebd['longitude'], station_name,
#                   station_coordinates, dst_min, ax=ax_map,
#                   distance_idx_nearest=distance_idx_nearest)
    
#     # Save figure if destination directory provided
#     # Change time format to avoid saving errors.
#     overpass_time_s = pd.Timestamp(time.item()).strftime('%d_%m_%Y_%H_%M_%S.%f')[:-4] 
#     if dstdir:
#         dstfile = f'{overpass_time_s}_L2_intercomparison_{keyword}.png'
#         fig.savefig(os.path.join(dstdir, dstfile), bbox_inches='tight', dpi=300)    
#     # Adjust layout
#     plt.tight_layout(rect=[0.1, 0.1, 0.88, 0.82])
#     fig.subplots_adjust(top=0.82, bottom=0.1, left=0.1, right=0.88)
    
#     return fig

# def plot_EC_L2_comparison(aebdpath, atcpath, sccfolderpath, pollypath, dstdir,
#                          resolution, fig_scale, network,
#                          max_distance=DEFAULT_CONFIG['MAX_DISTANCE'],
#                          hmax=DEFAULT_CONFIG['HMAX'], raman=True, klett=True,
#                          figsize=DEFAULT_CONFIG['FIGSIZE']):
#     """
#     Create comparison plots between EarthCARE L2 and ground-based data.
    
#     Parameters
#     ----------
#     aebdpath: str            | Path to AEBD product file
#     atcpath: str             | Path to ATC product file
#     sccfolderpath: str       | Path to SCC data folder
#     pollypath: str           | Path to PollyNET data file
#     dstdir: str              | Output directory for plots
#     resolution: str          | Resolution of data ('high', 'medium', 'low')
#     max_distance: float      | Maximum distance in km for data selection (def: 100)
#     hmax: float              | Maximum height for plots in meters (def: 16000)
#     lin_scale: bool          | Use linear scale for profiles (default: True)
#     log_scale: bool          | Use logarithmic scale for profiles (default: False)
#     figsize: tuple           | Figure size in inches (width, height) (def: (35,20))
    
#     Returns
#     -------
#     fig: matplotlib.figure   | The generated comparison plot figure
#     """
#     # Plot GND data
#     gnd_quicklook, gnd_profile, station_name, station_coordinates = load_ground_data(network,
#                                                                                      pollypath, 
#                                                                                      sccfolderpath, 'L2')
#     #Load and process EarthCARE products
#     # Load and crop AEBD  and  ATC product
#     aebd, aebd_50km, shortest_time, aebd_baseline, distance_idx_nearest, \
#         dst_min, s_dist_idx, aebd_100km = load_crop_EC_product(
#             aebdpath, station_coordinates, product='AEBD',
#             max_distance=max_distance, second_trim=True, second_distance=100)

#     # Load and crop AEBD  and  ATC product
#     atc, atc_100km, atc_baseline = load_crop_EC_product(
#         atcpath, station_coordinates, 'ATC', max_distance=100)

#     baseline = [aebd_baseline, atc_baseline]
    
#     # Format overpass date
#     overpass_date = pd.Timestamp(shortest_time.item()).strftime('%d-%m-%Y %H:%M')
#     overpass_date = '2023-09-09 03:10:20' #mock value for dummy  files.
    
#     if network == 'POLLYXT':
#         # Crop POLLY quicklook data around the overpass time
#         gnd_quicklook = crop_polly_file(gnd_quicklook ,overpass_date)

#     for idx in range(s_dist_idx-2, s_dist_idx+2):
#         for time_idx in range(gnd_profile.dims['time']):    
#             polly_raman, polly_klett = read_pollynet_profile(
#                   gnd_profile.isel(time=time_idx), data=True)
#             gnd_datasets = []
#             keywords = []  
#             if raman: 
#                     gnd_datasets.append(polly_raman)
#                     keywords.append('Raman')
#             if klett:
#                     gnd_datasets.append(polly_klett)   
#                     keywords.append('Klett')
                    
#             for i, (gnd_data, keyword) in enumerate(zip(gnd_datasets, keywords)):
                    
#                 plot_sub_L2(idx, resolution, gnd_quicklook, station_name,
#                            station_coordinates, aebd, aebd_50km,
#                            shortest_time, baseline, distance_idx_nearest,
#                            dst_min, s_dist_idx, aebd_100km, atc, atc_100km,
#                            gnd_data, dstdir, hmax, fig_scale, 
#                            network, keyword, figsize=figsize)
