#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Organization Module

Arranges the main plotting functions for the L1 and L2 visualization tools.

@author: Andreas Karipis
"""

import sys
sys.path.append('/home/akaripis/earthcare')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs

from ectools.ectools_bit import ecio, ecplot as ecplt, colormaps as clm
from valconfig import DEFAULT_CONFIG_L1, DEFAULT_CONFIG_L2
from valio import*
from valplot import*
import pdb

def plot_EC_L1_comparison(anompath, simpath, gndfolderpath,  dstdir, network, 
                          fig_scale, max_distance=DEFAULT_CONFIG_L1['MAX_DISTANCE'],
                          hmax=DEFAULT_CONFIG_L1['HMAX'], figsize=DEFAULT_CONFIG_L1['FIGSIZE']):
    
    """
    This function loads data from multiple sources (EarthCARE ATLID, simulator, 
    and ground station), processes it, and creates a multi-panel visualization 
    comparing the different measurements.
    
    Parameters
    ----------
    anompath : str                  |Path to the ANOM data file from EarthCARE
    simpath : str                   |Path to the simulator data file
    gndfolderpath : str             |Path to the folder containing ground station 
                                     data
    distdir : str                  |Directory where output figures will be saved
    network: str                   | Gnd data network's data that are processed
    max_distance : float, optional |Maximum distance in kilometers to consider for 
                                      nearby points, by default 50
    hmax : float, optional        |Maximum height for vertical axis in meters, 
                                    default 16000
    lin_scale : bool, optional    |Whether to use linear scale for profile plots, 
                                    by default True
    log_scale : bool, optional    |Whether to use logarithmic scale for profile 
                                   plots, by default False
    figsize : tuple, optional     |Figure size in inches (width, height), default 
                                    (27, 15)
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated comparison plot figure
        
    Notes
    -----
    The function creates a complex figure with multiple panels:
    - Left panels: Three quicklooks from ANOM data
    - Center panels: Two quicklooks from ground station
    - Right panels: Three profile comparisons and a map
    
    The figure is automatically saved if distdir is provided.
    """

    lin_scale = fig_scale != 'log'  # True unless fig_scale is 'log'
    log_scale = fig_scale != 'linear'  # True unless fig_scale is 'linear'
    # Load GND data
    gnd_quicklook, station_name, station_coordinates = load_ground_data(network,
                                                                        gndfolderpath,'L1')
    print('Loaded ground files, moving to simulator ')
    # Load simulator data - no preproccessing needed
    SIM = ecio.load_ANOM(simpath)
    SIM = SIM.isel(height=slice(None, None, -1))
    print('Loaded simulator file, moving to EC files ')

    # Load and crop EC product
    anom, anom_50km, shortest_time, baseline, distance_idx_nearest, dst_min, dist_idx = (
        load_crop_EC_product( anompath, station_coordinates, 'ANOM', max_distance=max_distance,
        second_trim=False, second_distance=100)
        )
    # Format overpass date
    overpass_date = pd.Timestamp(shortest_time.item()).strftime('%d-%m-%Y %H:%M')
    #overpass_date = '2023-09-24 14:10:20' #mock value for dummy  files.

    if network == 'POLLYXT' or network == 'EARLINET':
        overpass_date_g = pd.Timestamp(shortest_time.item())#.strftime('%d-%m-%Y %H:%M')
        #overpass_date = '2024-16-10 12:40:52' #mock value for dummy  files.

        gnd_quicklook = crop_polly_file(gnd_quicklook, overpass_date_g)
        
    print('File Loading successful')

    # Initialize the figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec( 9, 6,
        figure=fig,
        width_ratios=[1, 1, 0.005, 1.3, 1.3, 1.2],
        height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        hspace=1.8,
        wspace=0.55,
        top=0.82
    )
    
    # Add main title
    fig.suptitle(
        f'EarthCARE A-NOM ({baseline}) Comparison with simulated data\n'
        f' based on {station_name} measurements at {overpass_date} UTC\n',
        fontsize=24, weight='bold',  va='top', y=0.94)
    
    # Create subplots
    # Adjust the anom quicklook axis
    ax1 = fig.add_subplot(gs[0:3, 0:2])
    ax2 = fig.add_subplot(gs[3:6, 0:2])
    ax3 = fig.add_subplot(gs[6:9, 0:2])
    
    adjustments = {
        ax1: {'width_scale': 1.05,'height_scale':0.95},
        ax2: {'width_scale': 1.05},
        ax3: {'width_scale': 1.05}
    }

    for ax, params in adjustments.items():
        adjust_subplot_position(ax, **params)
        
    # Plot anom quicklooks
    ecplt.quicklook_ANOM(anom_50km, hmax=hmax[0], dstdir=dstdir,
                         axes=[ax1, ax2, ax3], comparison=True, 
                         station=shortest_time )
    
    # Adjust the scc quicklook axis
    ax4 = fig.add_subplot(gs[0:4, 3:4])
    ax5 = fig.add_subplot(gs[0:4, 4:5])

    adjustments = {
        ax4: {'x_offset': 0.003},
        ax5: {'x_offset': 0.003}
    }

    for ax, params in adjustments.items():
        adjust_subplot_position(ax, **params)

    # Plot GND data
    if network == 'EARLINET' or network == 'THELISYS':
        variables_q = ['range_corrected_signal', 'volume_linear_depolarization_ratio']
        titles_q = [f'{station_name} range.cor.signal', f'{station_name} vol.depol.ratio']
        plot_range = [[0, 15e8], [0, 0.2]]
        heightvar = 'altitude'
        units = ['m⁻¹ sr⁻¹','-']
    elif network == 'POLLYXT':
        variables_q = ['attenuated_backscatter_355nm', 'volume_depolarization_ratio_355nm']
        titles_q = [f'{station_name} att.bsc', f'{station_name} vol.depol.ratio']
        plot_range = [[0, 1.5e-5], [0, 0.3]]
        heightvar= 'height'
        units = ['m⁻¹ sr⁻¹','-']
    elif network == 'LICHT':
        variables_q = ['particle_backscatter_coefficient_355nm', 'volume_linear_depol_ratio_532nm']
        titles_q = [f'{station_name}  part. bsc coeff. 355 nm', f'{station_name} vol.depol.ratio 532 nm']
        plot_range = [[1e-7, 5e-5], [0.005, 0.8]]
        heightvar = 'height'
        units = ['m⁻¹ sr⁻¹','-']
    else:
        # Default case or error handling
        raise ValueError(f"Unsupported network: {network}. Must be either 'EARLINET' or 'POLLYXT'")
        
    axs = [ax4,ax5]
    
    for i, (ax, variable, title, p_range, unit) in enumerate(zip(axs, variables_q, titles_q, plot_range, units)):
        ecplt.plot_gnd_2D(ax, gnd_quicklook, variable, ' ', heightvar=heightvar,
                          cmap=clm.chiljet2, plot_scale='linear',
                          plot_range=p_range, units=unit, hmax=hmax[1],
                          plot_position='bottom',
                          title=title, comparison=True,
                          scc=True, yticks=(i == 0), xticks=False)
    # Adjust the profiles axis
    ax6 = fig.add_subplot(gs[4:, 3])
    ax7 = fig.add_subplot(gs[4:, 4])
    ax8 = fig.add_subplot(gs[4:, 5])

    adjustments = {
        ax6: {'height_scale': 0.96, 'width_scale': 0.96},
        ax7: {'height_scale': 0.96, 'width_scale': 0.96,'x_offset': -0.0015},
        ax8: {'height_scale': 0.96, 'width_scale': 0.96}
    }

    for ax, params in adjustments.items():
        adjust_subplot_position(ax, **params)
        
    # Define the profile axis ranges 
    xlims = DEFAULT_CONFIG_L1['DEFAULT_XLIMS'] if lin_scale else None
    xlims_log = DEFAULT_CONFIG_L1['DEFAULT_XLIMS_LOG'] if log_scale else None
    
    # Define the variables that will be plotted. Must be 3. 
    variables = DEFAULT_CONFIG_L1['VARIABLES']
    
    # Plot variable profiles
    plot_profile_comparison(anom_50km, SIM, variables, [ax6, ax7, ax8], 
                            hmax=hmax[2],xlim=xlims, xlim_log=xlims_log, 
                            lin_scale=lin_scale, log_scale=log_scale)

    # Adjust map plot axis
    ax9 = fig.add_subplot(gs[0:4, 5], projection=ccrs.PlateCarree())
    adjust_subplot_position( ax9,x_offset=0.01,y_offset=-0.04, height_scale=1.4,
                            width_scale=1.2)

    # Plot overpass map
    plot_orbit_map( anom['latitude'], anom['longitude'], station_name, station_coordinates,
        dst_min,ax=ax9, distance_idx_nearest=distance_idx_nearest,
        max_distance=DEFAULT_CONFIG_L1['MAX_DISTANCE'],idx=None)
    
    # Save figure if destination directory is provided
    if dstdir:
        dstfile = f'{overpass_date}_L1({baseline})_intercomparison.png'
        fig.savefig(f'{dstdir}/{dstfile}', bbox_inches='tight')
    
    # Adjust layout
    plt.tight_layout(rect=[0.1, 0.1, 0.88, 0.85])
    fig.subplots_adjust(top=0.82, bottom=0.1, left=0.1, right=0.88)
    
    return fig

def plot_sub_L2(idx, resolution, gnd_quicklooks, station_name, station_coordinates,
                      aebd, aebd_50km, shortest_time, baseline,
                      distance_idx_nearest, dst_min, aebd_profiles, atc,
                      atc_100km, gnd_profiles, dstdir, hmax,
                      fig_scale , network, keyword=None, figsize=(35, 20), 
                      smoothing = False):
    """
    Creates L2 comparison plots between EarthCARE and ground data.
    
    Parameters
    ----------
    idx: int                 | Index for profile selection
    resolution: str          | Data resolution ('high', 'medium', 'low')
    scc: xarray.Dataset      | SCC ground station data
    station_name: str        | Name of ground station
    station_coordinates: list| [latitude, longitude] of station
    aebd: xarray.Dataset     | Full AEBD dataset
    aebd_50km: xarray.Dataset| AEBD data within 50km of station
    shortest_time: datetime  | Time of closest approach
    baseline: str           | Processing baseline version
    distance_idx_nearest: array| Indices of nearby points
    dst_min: float          | Minimum distance to station
    aebd_100km: xarray.Dataset| AEBD data within 100km of station
    atc: xarray.Dataset      | Full ATC dataset
    atc_100km: xarray.Dataset| ATC data within 100km of station
    gnd_profiles: xarray.Dataset | Ground-based profile data
    hmax: float             | Maximum height for plots in meters (default: 16000)
    network: str             | Ground network, for processing the data
    keyword: str            | Type of ground data (default: 'Raman')
    figsize: tuple          | Figure size in inches (default: (35, 20))
    lin_scale: bool         | Use linear scale (default: True)
    log_scale: bool         | Use logarithmic scale (default: False)
    
    Returns
    -------
    fig: matplotlib.figure   | The generated comparison plot
    """

    lin_scale = fig_scale != 'log'  # True unless fig_scale is 'log'
    log_scale = fig_scale != 'linear'  # True unless fig_scale is 'linear'
        
    #pdb.set_trace()    
    time = (aebd_50km['time'])[idx]
    overpass_time = pd.Timestamp(time.item()).strftime('%d-%m-%Y %H:%M:%S.%f')[:-7]

    if network == 'LICHT':
        gnd_overpass_time = pd.Timestamp(gnd_profiles['time'].values.item()).strftime('%d-%m-%Y %H:%M:%S.%f')[:-4]
        gnd_overpass_time_fname = pd.Timestamp(gnd_profiles['time'].values.item()).strftime('%H_%M_%S')
        
    elif network == 'POLLYXT':
                # Extract start time
        start_timestamp = pd.to_datetime(gnd_profiles['start_time'].values.item(), unit='s')
        start_date = start_timestamp.strftime('%d-%m-%Y')
        start_hour = start_timestamp.strftime('%H:%M')
        
        # Extract end time
        end_timestamp = pd.to_datetime(gnd_profiles['end_time'].values.item(), unit='s')
        end_hour = end_timestamp.strftime('%H:%M')
        
        # Create the final combined format: DD_MM_YYYY HHMM_HHMM
        gnd_overpass_time = f"{start_date} {start_hour}-{end_hour}"
        gnd_ov_time_fname = start_timestamp.strftime('%H_%M')
    elif network == 'THELISYS':
        start_timestamp = pd.to_datetime(gnd_profiles['START_TIME'].values.item())
        start_date = start_timestamp.strftime('%d_%m_%Y')
        start_hour = start_timestamp.strftime('%H%M')
    
        # Extract end time - same approach
        end_timestamp = pd.to_datetime(gnd_profiles['END_TIME'].values.item())
        end_hour = end_timestamp.strftime('%H%M')
    
        # Create the final combined format: DD_MM_YYYY HHMM_HHMM
        gnd_overpass_time = f"{start_date} {start_hour}_{end_hour}"
        gnd_ov_time_fname =  start_timestamp.strftime('%H_%M')

    else:
            # Convert to a pandas Timestamp
        gnd_time = pd.to_datetime(gnd_profiles['time'].values.item(), unit='ns', origin='unix')
        
        # Format as needed
        gnd_overpass_time = gnd_time.strftime('%d_%m_%Y %H%M')
        gnd_ov_time_fname = gnd_time.strftime('%H_%M')


    # Initialize figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(10, 8, figure=fig, width_ratios=[1, 1, 1, 1.3, 1.3, 1.3, 1.3,
                 1.2], height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], hspace=1.8,
                 wspace=0.6, top=0.85)
    
    if network == 'LICHT':
        lidar_name = 'MPI LICHT' # PollyXT # THELISYS

        fig.suptitle(f'EarthCARE A-EBD({baseline[0]}) & A-TC({baseline[1]}) Comparison with '
                     f' {station_name} Ground Station - {lidar_name} L2 Retrieval \n'
                     f'ECA: {overpass_time} UTC - '
                     f'{lidar_name}: {gnd_overpass_time} UTC\n',
                     fontsize=26, weight='bold', va='top', y=.96)
    else:
        # Add main title
        fig.suptitle(f'EarthCARE A-EBD({baseline[0]}) & A-TC({baseline[1]}) Comparison at {overpass_time} UTC with\n'
                     f' {station_name} Ground Station L2 {keyword} Retrieval at {gnd_overpass_time} UTC',
                     fontsize=26, weight='bold', va='top', y=.96)


    # Create and adjust quicklook axes
    ax1 = fig.add_subplot(gs[0:2, 0:3])
    ax2 = fig.add_subplot(gs[2:4, 0:3])
    ax3 = fig.add_subplot(gs[4:6, 0:3])
    ax4 = fig.add_subplot(gs[6:8, 0:3])
    ax5 = fig.add_subplot(gs[8:10, 0:3])
    
    adjustments = {
        ax1: {'x_offset': -0.01, 'height_scale': 1},
        ax2: {'x_offset': -0.01, 'height_scale': 1},
        ax3: {'x_offset': -0.01, 'height_scale': 1},
        ax4: {'x_offset': -0.01, 'height_scale': 1},
        ax5: {'x_offset': -0.01, 'height_scale': 1.12}
    }

    for ax, params in adjustments.items():
        adjust_subplot_position(ax, **params)
    #pdb.set_trace()
    # Plot EBD and TC
    ecplt.quicklook_AEBD(aebd_50km, resolution=resolution, hmax=hmax[0], #hmax=1.5*hmax if hmax < 30e3 else 30e3,
                         dstdir=None, axes=[ax1, ax2, ax3, ax4, ax5],
                         comparison=True, station=shortest_time, show_temperature=True)
    
    ecplt.quicklook_ATC(atc_100km, hmax=hmax[0],# hmax=1.5*hmax if hmax < 30e3 else 30e3, 
                        resolution=resolution, dstdir=None, axes=ax5, 
                        comparison=True, station=shortest_time,show_temperature=True)
    
    # Create and adjust SCC axes
    ax6 = fig.add_subplot(gs[0:4, 3:5])
    ax7 = fig.add_subplot(gs[0:4, 5:7])
    
    adjustments = {
        ax6: {'x_offset': 0.01, 'width_scale': 0.95,'height_scale': 0.95,'y_offset': 0.01,},
        ax7: {'x_offset': 0, 'width_scale': 0.95,'height_scale': 0.95,'y_offset': 0.01,}
    }

    for ax, params in adjustments.items():
        adjust_subplot_position(ax, **params)

    # Plot GND data
    if network == 'EARLINET' or network == 'THELISYS':
        variables_q = ['range_corrected_signal', 'volume_linear_depolarization_ratio']
        titles_q = [f'{station_name} range.cor.signal', f'{station_name} vol.depol.ratio']
        plot_range = [[1e-6, 15e8], [1e-6, 0.2]]
        heightvar = 'altitude'
        units=['-','-']
    elif network == 'POLLYXT':
        variables_q = ['quasi_bsc_532', 'quasi_pardepol_532']
        titles_q = [f'{station_name} att.bsc', f'{station_name} par.depol.ratio']
        plot_range = [[0, 15e-6], [0, 0.4]]
        heightvar = 'height'
        units = ['m⁻¹ sr⁻¹','-']
    elif network == 'LICHT':
        variables_q = ['particle_backscatter_coefficient_355nm', 'volume_linear_depol_ratio_532nm']
        titles_q = [f'{station_name}  part. bsc coeff. 355 nm', f'{station_name} vol.depol.ratio 532 nm']
        plot_range = [[1e-7, 5e-5], [0.005, 0.8]]
        heightvar = 'height'
        units = ['m⁻¹ sr⁻¹','-']
    else:
        # Default case or error handling
        raise ValueError(f"Unsupported network: {network}. Must be either 'EARLINET' or 'POLLYXT'")
        
    axs = [ax6,ax7]
    
    for i, (ax, variable, title, p_range, unit) in enumerate(zip(axs, variables_q, titles_q, plot_range, units)):
        ecplt.plot_gnd_2D(ax, gnd_quicklooks, variable, ' ', heightvar=heightvar,
                           cmap=clm.chiljet2, plot_scale='linear', plot_range=p_range,
                           units=unit, hmax=hmax[1],# hmax=hmax if hmax < 22e3 else 22e3,
                          plot_position='bottom',
                          title=title, comparison=True,
                          scc=True, yticks=(i == 0), xticks=False)


    # Create and adjust profile axes
    ax8 = fig.add_subplot(gs[4:10, 3])
    ax9 = fig.add_subplot(gs[4:10, 4])
    ax10 = fig.add_subplot(gs[4:10, 5])
    ax11 = fig.add_subplot(gs[4:10, 6:7])
    if not DEFAULT_CONFIG_L2['COMP_TYPE'] == 'average':
        ax12 = fig.add_subplot(gs[4:10, 7])
    

    if not DEFAULT_CONFIG_L2['COMP_TYPE'] == 'average':
        adjustments = {
            ax8: {'x_offset': 0.05, 'height_scale': 1},
            ax9: {'x_offset': 0.04, 'height_scale': 1},
            ax10: {'x_offset': 0.02, 'height_scale': 1},
            ax11: {'x_offset': 0.01, 'height_scale': 1},
            ax12: {'height_scale': 1}
        }
    else: 
        adjustments = {
            ax8: {'x_offset': 0.06, 'height_scale': 1, 'width_scale':1.3},
            ax9: {'x_offset': 0.07, 'height_scale': 1,'width_scale':1.3},
            ax10: {'x_offset': 0.08, 'height_scale': 1,'width_scale':1.3},
            ax11: {'x_offset': 0.08, 'height_scale': 1,'width_scale':1.3}
        }
    
    for ax, params in adjustments.items():
        adjust_subplot_position(ax, **params)
    
    # Define variables and axes for profiles
    variables = [
        'particle_backscatter_coefficient_355nm',
        'particle_extinction_coefficient_355nm',
        'lidar_ratio_355nm',
        'particle_linear_depol_ratio_355nm'
    ]
    
    axes = [ax8, ax9, ax10, ax11]
    # Define the profile axis ranges 
    xlims = DEFAULT_CONFIG_L2['DEFAULT_XLIMS'] if lin_scale else None
    xlims_log = DEFAULT_CONFIG_L2['DEFAULT_XLIMS_LOG'] if log_scale else None
    
    # Plot AEBD profiles
    titles = ['Bsc. Coef.', 'Ext. Coef.', 'Lidar Ratio', 'Lin. depol. ratio']
    #pdb.set_trace()
    idxx=idx
    if DEFAULT_CONFIG_L2['COMP_TYPE'] == 'average':
        
        idx=slice(None)
        
    else:
        idx = idx
    # Plot ground data if available
    # gnd_profilesf = truncate_at_deviation(aebd_profiles, gnd_profiles, variables)
    for i, (variable, ax, title) in enumerate(zip(variables, axes, titles)):
        if variable in gnd_profiles:
            plot_AEBD_profiles(gnd_profiles, variable, ax=ax, lin_scale=lin_scale,
                             hmax=hmax[2],log_scale=log_scale, profile='GND',
                             yticks=(i == 0),smoothing=smoothing)  # Only True for first axis
        plot_AEBD_profiles(aebd_profiles, variable,hmax=hmax[2],resolution=resolution,
                           ax=ax, lin_scale=lin_scale,idx=idx,
                           log_scale=log_scale,title=title, profile='EC',
                           xlim=xlims[i] if xlims else None,
                           xlim_log=xlims_log[i] if xlims_log else None,
                           yticks=(i == 0))  # Only True for first axis
    
    if not DEFAULT_CONFIG_L2['COMP_TYPE'] == 'average':
        
        
        # Plot classification and quality status
        plot_AEBD_cla_qs(atc_100km, 'classification', 'quality_status', idx=idxx, 
                         resolution=resolution, hmax=hmax[2], title='Classification & \nQuality Status',
                         ax=ax12, yticks=False)
    # Create and adjust map plot
    ax_map = fig.add_subplot(gs[0:4, 7], projection=ccrs.PlateCarree())
    adjust_subplot_position(ax_map, x_offset=0.01, y_offset=-0.08,
                          height_scale=2.5, width_scale=1.4)

    # Plot map
    plot_orbit_map(aebd['latitude'], aebd['longitude'], station_name,
                  station_coordinates, dst_min, ax=ax_map,
                  distance_idx_nearest=distance_idx_nearest,
                  max_distance=DEFAULT_CONFIG_L2['MAX_DISTANCE'], idx=idxx,
                  lat2=aebd_50km['latitude'], lon2=aebd_50km['longitude'])
    
    # Save figure if destination directory provided
    # Change time format to avoid saving errors.
    overpass_time_s = pd.Timestamp(time.item()).strftime('%d_%m_%Y_%H_%M_%S.%f')[:-7]
    
    if DEFAULT_CONFIG_L2['COMP_TYPE'] == 'average':
        if network == 'LICHT':
            dstfile = f'{overpass_time_s}_gnd{gnd_overpass_time_fname}_L2_intercomparison_{keyword}.png'
        else:
            dstfile = f'{network}_{keyword}{gnd_ov_time_fname}_{resolution}_{DEFAULT_CONFIG_L2['MAX_DISTANCE']}_avg.png'
    else:
        if network == 'LICHT':
            dstfile = f'{overpass_time_s}_gnd{gnd_overpass_time_fname}_L2_intercomparison_{keyword}.png'
        else:
            dstfile = f'{network}_{keyword}{gnd_ov_time_fname}_{resolution}_{DEFAULT_CONFIG_L2['MAX_DISTANCE']}_{idx}.png'
    
    if dstdir:
        fig.savefig(os.path.join(dstdir, dstfile), bbox_inches='tight', dpi=300)    
    # Adjust layout
    plt.tight_layout(rect=[0.1, 0.1, 0.88, 0.82])
    fig.subplots_adjust(top=0.82, bottom=0.1, left=0.1, right=0.88)
    
    return fig

def plot_EC_L2_comparison(aebdpath, atcpath, gndfolderpath, dstdir, resolution, 
                          fig_scale, network, max_distance=DEFAULT_CONFIG_L2['MAX_DISTANCE'],
                          hmax=DEFAULT_CONFIG_L2['HMAX'], raman=True, klett=False,
                          figsize=DEFAULT_CONFIG_L2['FIGSIZE'], 
                          smoothing = DEFAULT_CONFIG_L2['SMOOTHING']):
    """
    Create comparison plots between EarthCARE L2 and ground-based data.
    
    Parameters
    ----------
    aebdpath: str            | Path to AEBD product file
    atcpath: str             | Path to ATC product file
    sccfolderpath: str       | Path to SCC data folder
    pollypath: str           | Path to PollyNET data file
    dstdir: str              | Output directory for plots
    resolution: str          | Resolution of data ('high', 'medium', 'low')
    max_distance: float      | Maximum distance in km for data selection (def: 100)
    hmax: float              | Maximum height for plots in meters (def: 16000)
    lin_scale: bool          | Use linear scale for profiles (default: True)
    log_scale: bool          | Use logarithmic scale for profiles (default: False)
    figsize: tuple           | Figure size in inches (width, height) (def: (35,20))
    
    Returns
    -------
    fig: matplotlib.figure   | The generated comparison plot figure
    """

    # Load and process GND data
    gnd_quicklook, gnd_profile, station_name, \
        station_coordinates = load_ground_data(network, gndfolderpath, 'L2',
                                                scc_term= 'b0355')
    ###Load and process EarthCARE products####
    
    # Load and crop AEBD  and  ATC product
    aebd, aebd_50km, shortest_time, aebd_baseline, distance_idx_nearest, \
        dst_min, s_dist_idx, aebd_100km = load_crop_EC_product(
            aebdpath, station_coordinates, product='AEBD',
            max_distance=max_distance, second_trim=True, second_distance=200)
    # Load and crop AEBD  and  ATC product
    atc, atc_100km, atc_baseline = load_crop_EC_product(
        atcpath, station_coordinates, 'ATC', max_distance=max_distance)
    #pdb.set_trace()
    baseline = [aebd_baseline, atc_baseline]
    
    #pdb.set_trace()
    
    overpass_date = pd.Timestamp(shortest_time.item()).strftime('%d-%m-%Y %H:%M')
    
    if network == 'POLLYXT' or network == 'EARLINET':
        overpass_date = pd.Timestamp(shortest_time.item())#.strftime('%d-%m-%Y %H:%M')
        #overpass_date = '2024-16-10 12:40:52' #mock value for dummy  files.
        if gnd_quicklook is not None:
            gnd_quicklook = crop_polly_file(gnd_quicklook, overpass_date)
        
    comp_type = DEFAULT_CONFIG_L2['COMP_TYPE']
    
    if comp_type == 'average':
        # idx_range = [s_dist_idx]
        # aebd_profile = aebd_50km.mean('along_track')
        idx_range = range(s_dist_idx-2, s_dist_idx+3)
        aebd_profile = aebd_50km.isel(along_track =idx_range).mean('along_track')
    elif comp_type == 'profile':
        idx_range = range(s_dist_idx-2, s_dist_idx+3)
        aebd_profile = aebd_50km
    
    raman = DEFAULT_CONFIG_L2['RETRIEVAL'] != 'KLETT'  # True unless RETRIEVAL is 'KLETT'
    klett = DEFAULT_CONFIG_L2['RETRIEVAL'] != 'RAMAN'  # True unless RETRIEVAL is 'RAMAN'

    for idx in idx_range:
        if network == 'EARLINET':
            # For EARLINET, gnd_profile is a list of datasets
            # Assuming gnd_profile[0] is raman and gnd_profile[1] is klett
            for time_idx in range(gnd_profile[0].dims['time']):  # Using first dataset for time dimension
                scc_raman, scc_klett = read_scc_profile(gnd_profile,time_idx)
                gnd_datasets = []
                keywords = []  
                if scc_raman is not None: 
                        gnd_datasets.append(scc_raman)
                        keywords.append('Raman')
                if scc_klett is not None:
                        gnd_datasets.append(scc_klett)   
                        keywords.append('Klett')
                for i, (gnd_data, keyword) in enumerate(zip(gnd_datasets, keywords)):
                    plot_sub_L2(idx, resolution, gnd_quicklook, station_name,
                                station_coordinates, aebd, aebd_50km,
                                shortest_time, baseline, distance_idx_nearest,
                                dst_min, aebd_profile, atc, atc_100km,
                                (gnd_data), dstdir, hmax, fig_scale, 
                                network, keyword,figsize=figsize,smoothing=smoothing)
        else:
         for time_idx in range(gnd_profile.dims['time']):    
            
            if network == 'LICHT':
                gnd_data = gnd_profile.isel(time=time_idx)
                plot_sub_L2(idx, resolution, gnd_quicklook, station_name,
                           station_coordinates, aebd, aebd_50km,
                           shortest_time, baseline, distance_idx_nearest,
                           dst_min, aebd_profile, atc, atc_100km,
                           gnd_data, dstdir, hmax, fig_scale, 
                           network, figsize=figsize,smoothing=smoothing)
            else:
                if network == 'POLLYXT':
                    polly_raman, polly_klett = read_pollynet_profile(gnd_profile.isel(time=time_idx), 
                                                                     data=True)
                elif network == 'THELISYS':
                    polly_raman, polly_klett = process_sula_profile(gnd_profile.isel(time=time_idx), 
                                                                    data=True)
                    
                gnd_datasets = []
                keywords = []  
                if raman: 
                        gnd_datasets.append(polly_raman)
                        keywords.append('Raman')
                if klett:
                        gnd_datasets.append(polly_klett)   
                        keywords.append('Klett')

                for i, (gnd_data, keyword) in enumerate(zip(gnd_datasets, keywords)):
                    
                    plot_sub_L2(idx, resolution, gnd_quicklook, station_name,
                           station_coordinates, aebd, aebd_50km,
                           shortest_time, baseline, distance_idx_nearest,
                           dst_min, aebd_profile, atc, atc_100km,
                           gnd_data, dstdir, hmax, fig_scale, 
                           network, keyword, figsize=figsize, smoothing=smoothing)