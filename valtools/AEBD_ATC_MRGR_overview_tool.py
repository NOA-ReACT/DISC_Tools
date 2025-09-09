#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EarthCARE Data Visualization Module

Creates comprehensive multi-panel visualizations for EarthCARE overpass data
including A-EBD, A-TC classification, M-RGR imagery, and orbit mapping.

@author: Andreas Karipis NOA-ReACT
"""

import sys
sys.path.append('/home/akaripis/earthcare')
sys.path.append('/home/akaripis/earthcare/valtools')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from matplotlib.colors import Normalize
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import matplotlib.patheffects as PathEffects
import cartopy.io.img_tiles as cimgt
import os
from datetime import datetime
import xarray as xr

# Import custom modules
from ectools.ectools_bit import ecio, ecplot as ecplt, colormaps as clm
from valio import *
from valplot import *



def plot_orbit_map(latitudes, longitudes, station_coordinates, station_coordinates1, 
                   buffer=1.4, ax=None, lat_min=None, lat_max=None, lon_min=None, 
                   lon_max=None, plot_features=True, plot_gridlines=True, 
                   plot_legend=True, use_satellite_imagery=False, 
                   distance_idx_nearest=None, zoom_level=11, output_dir=None, 
                   filename=None, dpi=300, format='png'):
    """
    Plot the satellite orbit path on a map with station locations.
    
    Parameters
    ----------
    latitudes: array-like         | Satellite latitudes
    longitudes: array-like        | Satellite longitudes
    station_coordinates: list     | [lat, lon] of primary station (PANGEA)
    station_coordinates1: list    | [lat, lon] of study area center
    buffer: float                 | Degrees to extend map boundaries (default: 1.4)
    ax: matplotlib.axes           | Axis to plot on. If None, creates new figure
    lat_min: float                | Custom map boundary minimum latitude
    lat_max: float                | Custom map boundary maximum latitude
    lon_min: float                | Custom map boundary minimum longitude
    lon_max: float                | Custom map boundary maximum longitude
    plot_features: bool           | Whether to add map features (default: True)
    plot_gridlines: bool          | Whether to add gridlines (default: True)
    plot_legend: bool             | Whether to show legend (default: True)
    use_satellite_imagery: bool   | Whether to use satellite imagery background (default: False)
    distance_idx_nearest: array   | Indices for plotted orbit segment
    zoom_level: int               | Zoom level for satellite imagery (default: 11)
    output_dir: str               | Directory to save figure
    filename: str                 | Filename for saved figure
    dpi: int                      | Resolution for saved figure (default: 300)
    format: str                   | File format for saved figure (default: 'png')
    
    Returns
    -------
    None                          | Displays plot and optionally saves figure
    """
    
    # Determine map boundaries
    if all(coord is not None for coord in [lat_min, lat_max, lon_min, lon_max]):
        map_lat_min, map_lat_max = lat_min, lat_max
        map_lon_min, map_lon_max = lon_min, lon_max
    else:
        # Center around satellite path
        lat_center = (np.max(latitudes) + np.min(latitudes)) / 2
        lon_center = (np.max(longitudes) + np.min(longitudes)) / 2
        map_lat_min = lat_center - 1.5 * buffer
        map_lat_max = lat_center + 1.5 * buffer
        map_lon_min = lon_center - buffer
        map_lon_max = lon_center + 1.2 * buffer
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 10),
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
    ax.set_extent([map_lon_min, map_lon_max, map_lat_min, map_lat_max],
                 crs=ccrs.PlateCarree())
    
    # Add satellite imagery if requested
    if use_satellite_imagery:
        try:
            imagery = cimgt.OSM()
            ax.add_image(imagery, zoom_level)
            print(f"Successfully loaded OpenStreetMap imagery")
        except Exception as e:
            print(f"Error loading OSM imagery: {e}")
            print("Falling back to standard map features.")
            use_satellite_imagery = False
                 
    # Plot orbit paths
    ax.plot(longitudes, latitudes, color='blue', lw=2,
            label='EC Orbit', transform=ccrs.PlateCarree())
    
    if distance_idx_nearest is not None:
        ax.plot(longitudes[distance_idx_nearest],
                latitudes[distance_idx_nearest],
                color='orangered', lw=2, label='Plotted orbit segment',
                transform=ccrs.PlateCarree())
    
    # Plot station markers
    if station_coordinates == station_coordinates1:
        ax.scatter(station_coordinates[1], station_coordinates[0],
                  color='red', s=150, label='PANGEA',
                  marker='*', transform=ccrs.PlateCarree())
    else: 
        ax.scatter(station_coordinates1[1], station_coordinates1[0],
                  color='green', s=180, label='Center of study orbit',
                  marker='o', transform=ccrs.PlateCarree())
        ax.scatter(station_coordinates[1], station_coordinates[0],
                  color='red', s=150, label='PANGEA',
                  marker='*', transform=ccrs.PlateCarree())
    
    # Add map features
    if plot_features:
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        if not use_satellite_imagery:
            ax.add_feature(cfeature.LAND, facecolor='#f5f2e8')
            ax.add_feature(cfeature.OCEAN, facecolor='#a8d5e2')
    
    # Add gridlines
    if plot_gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.2,
                         color='white' if use_satellite_imagery else 'black', 
                         alpha=0.8 if use_satellite_imagery else 0.6, 
                         linestyle='--')
        
        gl.top_labels = True
        gl.right_labels = True
        gl.left_labels = True
        gl.bottom_labels = True
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'size': 11, 'color': 'white' if use_satellite_imagery else 'black'}
        gl.ylabel_style = {'size': 11, 'color': 'white' if use_satellite_imagery else 'black'}
    
    # Show legend
    if plot_legend:
        ax.legend(loc='upper right', fontsize=10)
    
    # Save figure if output directory specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orbit_map_{timestamp}.{format}"
        elif not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {filepath}")
    
    plt.show()


def create_earthcare_gridspec(fig, figsize=(32, 16)):
    """
    Create standardized GridSpec layout for EarthCARE visualization.
    
    Parameters
    ----------
    fig: matplotlib.figure.Figure | The figure object to create GridSpec for
    figsize: tuple                | Figure size in inches (width, height) (default: (32, 16))
        
    Returns
    -------
    dict                          | Dictionary containing all subplot axes with descriptive keys
    gs: matplotlib.gridspec.GridSpec | The GridSpec object for further customization
    """
    
    # Create GridSpec: 5 rows, 13 columns with specific ratios
    gs = GridSpec(5, 13, figure=fig, 
                  width_ratios=[1, 1, 1, 0.005, 1, 1, 1, 1, 1, 0.1, 1.5, 2, 3],
                  height_ratios=[1, 1, 1, 1, 1], 
                  hspace=0.3,
                  wspace=0.8,
                  top=0.85)
    
    # Create subplot axes
    axes = {}
    
    # AEBD plots (first 3 columns, rows 0-3)
    axes['aebd1'] = fig.add_subplot(gs[0, 0:5])
    axes['aebd2'] = fig.add_subplot(gs[1, 0:5])
    axes['aebd3'] = fig.add_subplot(gs[2, 0:5])
    axes['aebd4'] = fig.add_subplot(gs[3, 0:5])
    
    # ATC plot (columns 5-8, rows 0-1)
    axes['atc'] = fig.add_subplot(gs[0:2, 5:9])
    
    # Map plot (columns 5-8, rows 2-3)
    axes['map'] = fig.add_subplot(gs[2:4, 5:9], projection=ccrs.PlateCarree())
    
    # MSI plot (column 11, rows 0-3)
    axes['msi'] = fig.add_subplot(gs[0:4, 11:12])
    
    return axes, gs


def adjust_subplot_position(ax, x_offset=0, y_offset=0, width_scale=1, height_scale=1):
    """
    Adjust subplot position and size.
    
    Parameters
    ----------
    ax: matplotlib.axes           | The subplot axis to adjust
    x_offset: float               | Horizontal position offset (default: 0)
    y_offset: float               | Vertical position offset (default: 0) 
    width_scale: float            | Width scaling factor (default: 1)
    height_scale: float           | Height scaling factor (default: 1)
    
    Returns
    -------
    None                          | Modifies axis position in place
    """
    pos = ax.get_position()
    new_pos = [pos.x0 + x_offset, pos.y0 + y_offset, 
               pos.width * width_scale, pos.height * height_scale]
    ax.set_position(new_pos)


def plot_earthcare_overview(aebd_50km, atc_100km, rgr_50km, lat, lon, hmax,
                           station_coordinates, station_coordinates1, 
                           shortest_time, distance_idx_nearest, 
                           overpass_datetime, figsize=(32, 16), 
                           output_dir=None):
    """
    Create comprehensive EarthCARE overview visualization.
    
    Parameters
    ----------
    aebd_50km: xr.Dataset         | A-EBD data within 50km of station
    atc_100km: xr.Dataset         | A-TC data within 100km of station
    rgr_50km: xr.Dataset          | M-RGR data within 50km of station
    lat: array-like               | Latitude coordinates
    lon: array-like               | Longitude coordinates
    station_coordinates: list     | [lat, lon] of primary station
    station_coordinates1: list    | [lat, lon] of study area center
    shortest_time: datetime       | Time of closest approach
    distance_idx_nearest: array   | Indices for plotted orbit segment
    overpass_datetime: datetime   | Overpass datetime for title
    figsize: tuple                | Figure size (default: (32, 16))
    output_dir: str               | Directory to save figure
        
    Returns
    -------
    fig: matplotlib.figure.Figure | The generated figure
    """
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=figsize)
    axes, gs = create_earthcare_gridspec(fig, figsize)
    
    # Format date for title
    try:
        formatted_date = overpass_datetime.strftime('%d-%m-%Y %H:%M')
    except:
        formatted_date = datetime.now().strftime('%d-%m-%Y %H:%M')
    
    # Add main title
    fig.suptitle(f'EarthCARE Overpass - {formatted_date}\n'
                 'A-EBD vs A-TC Classification and M-RGR overview',
                 fontsize=20, fontweight='bold', y=0.96)
    
    # Plot AEBD quicklook
    aebd_axes = [axes['aebd1'], axes['aebd2'], axes['aebd3'], axes['aebd4']]
    quicklook_aebd = ecplt.quicklook_AEBD(aebd_50km, resolution='low', hmax=hmax, 
                                          dstdir=None, comparison=True, 
                                          station=shortest_time, show_temperature=True,
                                          axes=aebd_axes)
    
    # Plot ATC quicklook
    quicklook_atc = ecplt.quicklook_ATC(atc_100km, resolution='low', hmax=hmax, 
                                        dstdir=None, comparison=True,
                                        station=shortest_time, axes=axes['atc'],
                                        show_temperature=True)
    
    # Plot orbit map
    plot_orbit_map(lat, lon, station_coordinates1, station_coordinates, 
                   lat_min=30, lat_max=45, lon_min=0, lon_max=35, 
                   ax=axes['map'], distance_idx_nearest=distance_idx_nearest, 
                   output_dir=None, filename="map", use_satellite_imagery=False)
    
    # Plot MSI data (RGB for day, TIR for night)
    axes['msi'].set_xlim(auto=True)
    axes['msi'].set_ylim(auto=True)
    
    if 8 <= overpass_datetime.hour < 18:
        rgb = ecplt.calculate_RGB_MRGR(rgr_50km)
        ecplt.plot_RGB_MRGR_vertical_minus90(axes['msi'], rgb, rgr_50km)
        ecplt.format_MRGR(axes['msi'], rgr_50km, orientation='vertical')
    else:
        ecplt.plot_TIR_MRGR_vertical_minus90(axes['msi'], rgr_50km)
        ecplt.format_MRGR(axes['msi'], rgr_50km, orientation='vertical')
    
    # Apply fine adjustments
    adjust_subplot_position(axes['atc'], x_offset=0.03)
    adjust_subplot_position(axes['map'], x_offset=0.03, y_offset=-0.07, 
                           width_scale=1.2, height_scale=1.5)
    
    # Set font sizes
    title_size = 18
    ylabel_size = 12
    ytick_size = 10
    xtick_size = 10
    
    all_axes = [axes['aebd1'], axes['aebd2'], axes['aebd3'], 
                axes['aebd4'], axes['atc'], axes['msi']]
    
    for ax in all_axes:
        ax.title.set_fontsize(title_size)
    
    # Set specific font sizes for AEBD axes
    for ax in aebd_axes:
        ax.yaxis.label.set_fontsize(ylabel_size)
        ax.tick_params(axis='y', labelsize=ytick_size)
    
    # Set x-tick sizes
    axes['aebd4'].tick_params(axis='x', labelsize=xtick_size)
    axes['atc'].tick_params(axis='x', labelsize=xtick_size)
    
    # Save figure
    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'combined_AEBD_ATC_map_{timestamp}.png'
        outpath = os.path.join(output_dir, filename)
        fig.savefig(outpath, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {outpath}")
    
    plt.show()
    return fig


def main():
    """
    Main execution function for EarthCARE visualization.
    
    Returns
    -------
    None                          | Executes visualization workflow
    """
    
    # Configuration
    root_dir = '/home/akaripis/earthcare/files/20250812/'
    paths = build_paths(root_dir, 'POLLYXT', 'L2')
    
    # Station coordinates
    station_coordinates = [35.86, 23.31]  # AKY
    station_coordinates1 = [35.86, 23.31]  # Same as primary
    cropping_dist = 600 # in km scale
    hmax =25e3
    # Load and crop EarthCARE products
    try:
        aebd, aebd_50km, shortest_time, aebd_baseline, distance_idx_nearest, \
            dst_min, s_dist_idx = load_crop_EC_product(
                paths['AEBD'], station_coordinates1, product='AEBD',
                max_distance=cropping_dist)
        
        atc, atc_100km, atc_baseline = load_crop_EC_product(
            paths['ATC'], station_coordinates1, 'ATC', max_distance=cropping_dist)
        
        rgr, rgr_50km, rgr_baseline = load_crop_EC_product(
            paths['MRGR'], station_coordinates1, product='MRGR',
            max_distance=cropping_dist)
        
        # Extract coordinates
        lat = aebd['latitude']
        lon = aebd['longitude']
        
        # Determine overpass datetime
        if 'shortest_time' in locals():
            overpass_datetime = pd.Timestamp(shortest_time.item())
        elif hasattr(aebd, 'time') and len(aebd.time) > 0:
            overpass_datetime = pd.Timestamp(aebd.time[0].item())
        else:
            overpass_datetime = datetime.now()
        
        # Create visualization
        fig = plot_earthcare_overview(
            aebd_50km, atc_100km, rgr_50km, lat, lon,hmax,
            station_coordinates, station_coordinates1,
            shortest_time, distance_idx_nearest, overpass_datetime,
            output_dir=paths.get('OUTPUT', './'))
        
        print("EarthCARE visualization completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()