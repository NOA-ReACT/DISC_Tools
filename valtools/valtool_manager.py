#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Organization Module

Arranges the main plotting functions for the L1 and L2 visualization tools.

@author: Andreas Karipis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> ectools_noa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs

# from ectools.ectools_bit import ecio, ecplot as ecplt, colormaps as clm
from ectools_noa import ecio, ecplot_standalone_v2 as ecplt, colormaps as clm
from valconfig import DEFAULT_CONFIG_L1, DEFAULT_CONFIG_L2
from valio import*
from valplot import*

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

    # Load and crop ANOM product
    try:
        anom, anom_50km, shortest_time, baseline, distance_idx_nearest, dst_min, dist_idx = (
            load_crop_EC_product( anompath, station_coordinates, 'ANOM', max_distance=max_distance,
            second_trim=False, second_distance=100)
            )
    except Exception as e:
        print("ANOM dataset not in range of the station.")
        raise
    
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

    
    if network in ('EARLINET', 'THELISYS'):
        variables_q  = ['range_corrected_signal', 'volume_linear_depolarization_ratio']
        titles_q     = [f'{station_name} range.cor.signal', f'{station_name} vol.depol.ratio']
        plot_scales  = ['log', 'linear']                      # first log, second linear
        plot_ranges  = [[1e7, 1e9], [0.0, 0.2]]            # numeric limits
        heightvar    = 'altitude'
        units        = ['m⁻¹ sr⁻¹', '-']
    
    elif network == 'POLLYXT':
        variables_q  = ['attenuated_backscatter_355nm', 'volume_depolarization_ratio_355nm']
        titles_q     = [f'{station_name} att.bsc', f'{station_name} vol.depol.ratio']
        plot_scales  = ['log', 'linear']
        plot_ranges  = [[1e-8, 3e-5], [0.0, 0.3]]
        heightvar    = 'height'
        units        = ['m⁻¹ sr⁻¹', '-']
    
    elif network == 'LICHT':
        variables_q  = ['particle_backscatter_coefficient_355nm', 'volume_linear_depol_ratio_532nm']
        titles_q     = [f'{station_name} part. bsc coeff. 355 nm', f'{station_name} vol.depol.ratio 532 nm']
        plot_scales  = ['log', 'linear']
        plot_ranges  = [[1e-7, 5e-5], [0.005, 0.8]]
        heightvar    = 'height'
        units        = ['m⁻¹ sr⁻¹', '-']
    
    else:
        raise ValueError(f"Unsupported network: {network}. Must be one of 'EARLINET', 'THELISYS', 'POLLYXT', 'LICHT'.")

    axs = [ax4, ax5]
    #pdb.set_trace()
    for i, (ax, variable, title, scale, p_range, unit) in enumerate(zip(axs, variables_q, titles_q, plot_scales, plot_ranges, units)):
           
        ecplt.plot_gnd_2D(ax, gnd_quicklook, variable, ' ', heightvar=heightvar,
                               cmap=clm.chiljet2, plot_scale=scale, plot_range=p_range,
                               units=unit, hmax=hmax[1],# hmax=hmax if hmax < 22e3 else 22e3,
                              plot_position='bottom',
                              title=title, comparison=True,
                              gnd=True, yticks=(i == 0), xticks=False)
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
                      fig_scale , network, keyword=None, idx_range=None, hmin=0, figsize=(35, 20), 
                      smoothing = False, comp_type='average'):
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
    hmin: float             | Profiles: data below hmin (m) is blanked (default: 0)
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
        
    time = (aebd_50km['time'])[idx]
    overpass_time = pd.Timestamp(time.item()).strftime('%d-%m-%Y %H:%M:%S.%f')[:-7]
    gnd_t0 = gnd_t1 = None

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

    # ground averaging window (shaded in the GND quicklooks)
    if network in ('POLLYXT', 'THELISYS'):
        gnd_t0, gnd_t1 = start_timestamp, end_timestamp

    # Initialize figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(10, 8, figure=fig, width_ratios=[1, 1, 1, 1.3, 1.3, 1.3, 1.3,
                 1.2], height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], hspace=1.8,
                 wspace=0.6, top=0.85)
    
    if network == 'POLLYXT' or network == 'EARLINET':
        # Add main title
        fig.suptitle(f'EarthCARE A-EBD({baseline[0]}) & A-TC({baseline[1]}) Comparison at {overpass_time} UTC with\n'
                      f' {station_name} Ground Station L2 {keyword} Retrieval at {gnd_overpass_time} UTC',
                      fontsize=26, weight='bold', va='top', y=.96)


    else:
        lidar_name = 'MPI LICHT' # PollyXT # THELISYS

        # fig.suptitle(f'EarthCARE A-EBD({baseline[0]}) & A-TC({baseline[1]}) Comparison with '
        #              f' {station_name} Ground Station - {lidar_name} L2 Retrieval \n'
        #              f'ECA: {overpass_time} UTC - '
        #              f'{lidar_name}: {gnd_overpass_time} UTC\n',
        #              fontsize=26, weight='bold', va='top', y=.96)


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
    # Plot EBD and TC
    ecplt.quicklook_AEBD(aebd_50km, resolution=resolution, hmax=hmax[0], #hmax=1.5*hmax if hmax < 30e3 else 30e3,
                          dstdir=None, axes=[ax1, ax2, ax3, ax4, ax5],
                          comparison=True, station=shortest_time, show_temperature=True)

    ecplt.quicklook_ATC(atc_100km, hmax=hmax[0],# hmax=1.5*hmax if hmax < 30e3 else 30e3, 
                        resolution=resolution, dstdir=None, axes=ax5, 
                        comparison=True, station=shortest_time,show_temperature=True)

    # mark the along-track profile(s) feeding the profile panels
    shade_ec = (idx_range is not None
                and len(idx_range) < aebd_50km.sizes['along_track'])
    if not shade_ec:
        for ax in (ax1, ax2, ax3, ax4, ax5):
            ax.axvline(time.values, color='black', lw=1.5, alpha=0.8, zorder=5)
    else:
        t_track = aebd_50km['time'].values
        sel = np.atleast_1d(idx_range)
        edges = np.concatenate([t_track[:1], t_track[:-1] + (t_track[1:] - t_track[:-1]) / 2, t_track[-1:]])
        for ax in (ax1, ax2, ax3, ax4, ax5):
            ax.axvspan(edges[sel[0]], edges[sel[-1] + 1],
                       facecolor='black', alpha=0.10, zorder=4)
            ax.axvline(edges[sel[0]],      color='red', ls=':', lw=2, zorder=5)
            ax.axvline(edges[sel[-1] + 1], color='red', ls=':', lw=2, zorder=5)
    
    # Create and adjust SCC axes
    ax6 = fig.add_subplot(gs[0:4, 3:5])
    ax7 = fig.add_subplot(gs[0:4, 5:7])
    
    adjustments = {
        ax6: {'x_offset': 0.01, 'width_scale': 0.95,'height_scale': 0.95,'y_offset': 0.01,},
        ax7: {'x_offset': 0, 'width_scale': 0.95,'height_scale': 0.95,'y_offset': 0.01,}
    }

    for ax, params in adjustments.items():
        adjust_subplot_position(ax, **params)

    # pick a tiny positive floor for log axes

    
    if network in ('EARLINET', 'THELISYS'):
        variables_q  = ['range_corrected_signal', 'volume_linear_depolarization_ratio']
        titles_q     = [f'{station_name} range.cor.signal', f'{station_name} vol.depol.ratio']
        plot_scales  = ['log', 'linear']                      # first log, second linear
        plot_ranges  = [[1e7, 1e9], [0.0, 0.4]]            # numeric limits
        heightvar    = 'altitude'
        units        = ['m⁻¹ sr⁻¹', '-']
    
    elif network == 'POLLYXT':
        variables_q = ['quasi_bsc_532', 'quasi_pardepol_532']
        titles_q = [f'{station_name} att.bsc', f'{station_name} par.depol.ratio']
        plot_scales  = ['log', 'linear']                      # first log, second linear
        plot_ranges = [[1e-8, 15e-6], [0, 0.4]]
        heightvar = 'height'
        units = ['m⁻¹ sr⁻¹','-']
    
    # elif network == 'LICHT':
    #     variables_q  = ['particle_backscatter_coefficient_355nm', 'volume_linear_depol_ratio_532nm']
    #     titles_q     = [f'{station_name} part. bsc coeff. 355 nm', f'{station_name} vol.depol.ratio 532 nm']
    #     plot_scales  = ['log', 'linear']
    #     plot_ranges  = [[1e-7, 5e-5], [0.005, 0.8]]
    #     heightvar    = 'height'
    #     units        = ['m⁻¹ sr⁻¹', '-']
    
    else:
        raise ValueError(f"Unsupported network: {network}. Must be one of 'EARLINET', 'THELISYS', 'POLLYXT', 'LICHT'.")

    axs = [ax6, ax7]

    for i, (ax, variable, title, scale, p_range, unit) in enumerate(zip(axs, variables_q, titles_q, plot_scales, plot_ranges, units)):
           
        ecplt.plot_gnd_2D(ax, gnd_quicklooks, variable, ' ', heightvar=heightvar,
                               cmap=clm.chiljet2, plot_scale=scale, plot_range=p_range,
                               units=unit, hmax=hmax[1],# hmax=hmax if hmax < 22e3 else 22e3,
                              plot_position='bottom',
                              title=title, comparison=True,
                              gnd=True, yticks=(i == 0), xticks=False)

    # mark the satellite overpass and the ground averaging window
    for ax in (ax6, ax7):
        ax.axvline(shortest_time, color='black', ls='--', lw=1.5, zorder=5)
        if gnd_t0 is not None:
            ax.axvspan(gnd_t0, gnd_t1, color='black', alpha=0.12, zorder=4)
            ax.axvline(gnd_t0, color='black', ls=':', lw=1.2, zorder=5)
            ax.axvline(gnd_t1, color='black', ls=':', lw=1.2, zorder=5)

    # Create and adjust profile axes
    ax8 = fig.add_subplot(gs[4:10, 3])
    ax9 = fig.add_subplot(gs[4:10, 4])
    ax10 = fig.add_subplot(gs[4:10, 5])
    ax11 = fig.add_subplot(gs[4:10, 6:7])
    if comp_type not in ['average', 'average_profiles']:
        ax12 = fig.add_subplot(gs[4:10, 7])
    
    if comp_type not in ['average', 'average_profiles']:
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
    if comp_type in ['average', 'average_profiles', 'profile']:
        idx=slice(None)
        
    else:
        idx = idx
    # Plot ground data if available
    # gnd_profilesf = truncate_at_deviation(aebd_profiles, gnd_profiles, variables)
    for i, (variable, ax, title) in enumerate(zip(variables, axes, titles)):
        if variable in gnd_profiles:
            plot_AEBD_profiles(gnd_profiles, variable, ax=ax, lin_scale=lin_scale,
                             hmax=hmax[2], hblank=hmin, log_scale=log_scale, profile='GND',
                             yticks=(i == 0),smoothing=smoothing)  # Only True for first axis
        plot_AEBD_profiles(aebd_profiles, variable,hmax=hmax[2], hblank=hmin, resolution=resolution,
                           ax=ax, lin_scale=lin_scale,idx=idx,
                           log_scale=log_scale,title=title, profile='EC',
                           xlim=xlims[i] if xlims else None,
                           xlim_log=xlims_log[i] if xlims_log else None,
                           yticks=(i == 0))  # Only True for first axis
    
    if comp_type not in ['average', 'average_profiles']:
        # Plot classification and quality status
        plot_AEBD_cla_qs(atc_100km, 'classification', 'quality_status', idx=idxx, 
                          resolution=resolution, hmax=hmax[2], title='Classification & \nQuality Status',
                          ax=ax12, yticks=False)
        # atmospheric_segments = [
        # # (0, 4266, 0, 'purple', 'Clear'),
        # (1100, 3880, 1, 'orange', 'CNS/FCA')
        # # (4864, 20000, 0, 'purple', 'Clear')
        # #(10000, 20000, 3, 'purple', 'Stratosphere (>10km)')
        # ]
        
        # # Call your function
        # plot_AEBD_cla_qs_manual(atc_100km, 'classification', idx=idxx, 
        #                   resolution=resolution, hmax=hmax[2], title='EC & Ground Classification',
        #                   ax=ax12, yticks=False, qs_segments=atmospheric_segments)
    # Create and adjust map plot
    ax_map = fig.add_subplot(gs[0:4, 7], projection=ccrs.PlateCarree())
    #ax_map.set_aspect('auto')  # Allow free aspect ratio
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
    
    if comp_type == 'average':
        if network == 'LICHT':
            dstfile = f'{overpass_time_s}_gnd{gnd_overpass_time_fname}_L2_intercomparison_{keyword}.png'
        else:
            dstfile = f'{network}_{keyword}{gnd_ov_time_fname}_{resolution}_{DEFAULT_CONFIG_L2['MAX_DISTANCE']}_avg_{baseline[0]}.png'
    elif comp_type =='average_profiles':
        if network == 'LICHT':
            dstfile = f'{overpass_time_s}_gnd{gnd_overpass_time_fname}_L2_intercomparison_{keyword}.png'
        else:
            dstfile = f'{network}_{keyword}{gnd_ov_time_fname}_{resolution}_{DEFAULT_CONFIG_L2['MAX_DISTANCE']}_avg_prof_{baseline[0]}.png'
    else:
        if network == 'LICHT':
            dstfile = f'{overpass_time_s}_gnd{gnd_overpass_time_fname}_L2_intercomparison_{keyword}.png'
        else:
            dstfile = f'{network}_{keyword}{gnd_ov_time_fname}_{resolution}_{DEFAULT_CONFIG_L2['MAX_DISTANCE']}_{idxx}_{baseline[0]}.png'
       
    # figure legend for the quicklook markers
    handles = [Line2D([], [], color='black', ls='--', lw=1.5,
                      label='closest satellite approach to station')]
    if shade_ec:
        handles.append(Patch(facecolor='black', alpha=0.10, edgecolor='black', ls=':',
                             label='profiles averaged for comparison'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 0.04), fontsize=18)

    if dstdir:
        fig.savefig(os.path.join(dstdir, dstfile), bbox_inches='tight', dpi=300)    
        
    # Adjust layout
    plt.tight_layout(rect=[0.1, 0.1, 0.88, 0.82])
    fig.subplots_adjust(top=0.82, bottom=0.1, left=0.1, right=0.88)
    
    return fig


def compute_layer_stats(ec_prof, gnd_prof, variable, hvar='JSG_height',
                        layer_km=(4.9, 6.0)):
    """
    Layer-mean agreement statistics (bias, relative bias, RMSD) between an EC
    profile and a ground profile over a height band given in KILOMETRES.

    Unit-aware: if the height coordinate spans > ~100 it is treated as metres
    and the km band is scaled accordingly. The ground profile is linearly
    interpolated onto the EC height grid before differencing.

    Returns a dict: ec_mean, gnd_mean, bias, rel_bias_pct, rmsd, n.
    NaN-filled if a profile is missing or the band is empty. Never raises.
    """
    import numpy as np
    nan = float('nan')
    empty = dict(ec_mean=nan, gnd_mean=nan, bias=nan,
                 rel_bias_pct=nan, rmsd=nan, n=0)
    try:
        if ec_prof is None or variable not in ec_prof:
            return empty
        # resolve EC height coord
        hv = hvar
        if hv not in getattr(ec_prof, 'coords', {}) and hv not in getattr(ec_prof, 'variables', {}):
            for c in ('JSG_height', 'height', 'altitude'):
                if c in ec_prof.coords or c in ec_prof.variables:
                    hv = c
                    break
        ec_v = np.asarray(ec_prof[variable].squeeze().values, dtype=float)
        z_ec = np.asarray(ec_prof[hv].squeeze().values, dtype=float)
        if z_ec.ndim != 1 or ec_v.shape != z_ec.shape:
            return empty
        scale = 1000.0 if np.nanmax(np.abs(z_ec)) > 100.0 else 1.0
        lo, hi = layer_km[0] * scale, layer_km[1] * scale
        m = (z_ec >= lo) & (z_ec <= hi)
        if not np.any(m):
            return empty
        ec_band = ec_v[m]
        z_band = z_ec[m]
        ec_mean = np.nanmean(ec_band)

        gnd_band = None
        if gnd_prof is not None and variable in gnd_prof:
            gv = 'height'
            for c in ('height', 'altitude', 'JSG_height'):
                if c in gnd_prof.coords or c in getattr(gnd_prof, 'variables', {}):
                    gv = c
                    break
            z_g = np.asarray(gnd_prof[gv].squeeze().values, dtype=float)
            g_raw = np.asarray(gnd_prof[variable].squeeze().values, dtype=float)
            if z_g.shape == g_raw.shape and z_g.ndim == 1:
                order = np.argsort(z_g)
                gnd_band = np.interp(z_band, z_g[order], g_raw[order],
                                     left=np.nan, right=np.nan)

        if gnd_band is None or np.all(np.isnan(gnd_band)):
            return dict(ec_mean=ec_mean, gnd_mean=nan, bias=nan,
                        rel_bias_pct=nan, rmsd=nan,
                        n=int(np.sum(~np.isnan(ec_band))))

        good = ~np.isnan(ec_band) & ~np.isnan(gnd_band)
        if not np.any(good):
            return dict(ec_mean=ec_mean, gnd_mean=nan, bias=nan,
                        rel_bias_pct=nan, rmsd=nan, n=0)
        gnd_mean = np.nanmean(gnd_band[good])
        bias = ec_mean - gnd_mean
        # relative bias is only meaningful when the ground mean is well above
        # the noise floor; otherwise dividing by a near-zero value produces
        # garbage (e.g. +1696 %). Gate on a fraction of the band's own scale.
        denom_scale = np.nanmax(np.abs(gnd_band[good])) if np.any(good) else 0.0
        if abs(gnd_mean) > 0.05 * denom_scale and gnd_mean != 0:
            rel = 100.0 * bias / gnd_mean
        else:
            rel = nan
        rmsd = np.sqrt(np.nanmean((ec_band[good] - gnd_band[good]) ** 2))
        return dict(ec_mean=ec_mean, gnd_mean=gnd_mean, bias=bias,
                    rel_bias_pct=rel, rmsd=rmsd, n=int(good.sum()))
    except Exception as _e:
        print('[cols] compute_layer_stats error:', repr(_e))
        return empty


def plot_sub_L2_columns(idx, resolution, gnd_quicklooks, station_name,
                        station_coordinates, aebd, aebd_50km, shortest_time,
                        baseline, distance_idx_nearest, dst_min, aebd_profiles,
                        atc, atc_100km, gnd_profiles, dstdir, hmax, fig_scale,
                        network, keyword=None, figsize=(32, 18),
                        smoothing=False, comp_type='average',
                        layer_km=(4.9, 6.0)):
    """
    Column-of-questions L2 comparison figure.

    Sibling to plot_sub_L2. EVERY ectools / valplot call below is identical to
    the corresponding call in plot_sub_L2 -- same function, same arguments.
    Only the GridSpec geometry differs, plus two NEW panels (bias dot-plot and
    stats table) that do not replace anything in the parent.

    GridSpec(4, 10):
        scene curtains : rows 0-4, cols 0-3   (AEBD stack + ATC on bottom axis,
                                               exactly as parent: 5 axes)
        info text      : (overlaid top-left, optional)
        beta / alpha / S / delta : cols 4,5,6,7  (one profile each, full height)
        by how much    : rows 0-2, cols 8-9
        stats table    : rows 2-4, cols 8-9
        map            : inset, top-left of scene region
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    import matplotlib.lines as mlines
    import cartopy.crs as ccrs

    lin_scale = fig_scale != 'log'
    log_scale = fig_scale != 'linear'

    # ----- timestamps : copied from plot_sub_L2 ----------------------------
    time = (aebd_50km['time'])[idx]
    overpass_time = pd.Timestamp(time.item()).strftime('%d-%m-%Y %H:%M:%S.%f')[:-7]

    if network == 'POLLYXT':
        start_timestamp = pd.to_datetime(gnd_profiles['start_time'].values.item(), unit='s')
        start_date = start_timestamp.strftime('%d-%m-%Y')
        start_hour = start_timestamp.strftime('%H:%M')
        end_timestamp = pd.to_datetime(gnd_profiles['end_time'].values.item(), unit='s')
        end_hour = end_timestamp.strftime('%H:%M')
        gnd_overpass_time = f"{start_date} {start_hour}-{end_hour}"
        gnd_ov_time_fname = start_timestamp.strftime('%H_%M')
    elif network == 'THELISYS':
        start_timestamp = pd.to_datetime(gnd_profiles['START_TIME'].values.item())
        start_date = start_timestamp.strftime('%d_%m_%Y')
        start_hour = start_timestamp.strftime('%H%M')
        end_timestamp = pd.to_datetime(gnd_profiles['END_TIME'].values.item())
        end_hour = end_timestamp.strftime('%H%M')
        gnd_overpass_time = f"{start_date} {start_hour}_{end_hour}"
        gnd_ov_time_fname = start_timestamp.strftime('%H_%M')
    elif network == 'LICHT':
        gnd_overpass_time = pd.Timestamp(gnd_profiles['time'].values.item()).strftime('%d-%m-%Y %H:%M:%S.%f')[:-4]
        gnd_ov_time_fname = pd.Timestamp(gnd_profiles['time'].values.item()).strftime('%H_%M_%S')
    else:
        gnd_time = pd.to_datetime(gnd_profiles['time'].values.item(), unit='ns', origin='unix')
        gnd_overpass_time = gnd_time.strftime('%d_%m_%Y %H%M')
        gnd_ov_time_fname = gnd_time.strftime('%H_%M')

    # keep the scalar index for the map marker / classification / filename,
    # before idx is later reassigned to slice(None) for the profile loop.
    idxx = idx

    # ----- figure scaffold : agreed geometry -------------------------------
    # 4 rows x 9 cols. Col 3 is a thin spacer between the scene block and the
    # profiles. Top strip (map+info) = rows 0-1; single scene curtain = rows
    # 2-3, both over cols 0-2. Profiles in cols 4-7. Right rail in col 8.
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 9, figure=fig,
                  width_ratios=[1, 1, 1, 0.15, 1.25, 1.25, 1.25, 1.25, 1.1],
                  height_ratios=[1] * 4,
                  hspace=0.55, wspace=0.55, top=0.85)

    if network in ('POLLYXT', 'EARLINET'):
        fig.suptitle(
            f'EarthCARE A-EBD({baseline[0]}) & A-TC({baseline[1]}) vs '
            f'{station_name} {keyword} \u2014 collocated validation\n'
            f'EC {overpass_time} UTC  \u00b7  GND {gnd_overpass_time} UTC',
            fontsize=20, weight='bold', va='top', y=0.965)

    # =======================================================================
    # TOP STRIP  (rows 0-1, cols 0-2) : map LEFT + info RIGHT
    # =======================================================================
    ax_map = fig.add_subplot(gs[0:2, 0], projection=ccrs.PlateCarree())
    adjust_subplot_position(ax_map, x_offset=0.0, y_offset=-0.01,
                            height_scale=1.35, width_scale=1.5)
    plot_orbit_map(aebd['latitude'], aebd['longitude'], station_name,
                   station_coordinates, dst_min, ax=ax_map,
                   distance_idx_nearest=distance_idx_nearest,
                   max_distance=DEFAULT_CONFIG_L2['MAX_DISTANCE'], idx=idxx,
                   lat2=aebd_50km['latitude'], lon2=aebd_50km['longitude'])

    ax_info = fig.add_subplot(gs[0:2, 1:3])
    ax_info.axis('off')
    info = (
        "Where & when\n"
        f"  station        {station_name}\n"
        f"  min. distance  {dst_min:.1f} km\n\n"
        "Timing\n"
        f"  EC   {overpass_time} UTC\n"
        f"  GND  {gnd_overpass_time} UTC\n\n"
        "Products\n"
        f"  EC   A-EBD({baseline[0]}) / A-TC({baseline[1]})\n"
        f"  GND  {network} L2 {keyword}\n\n"
        "Matching\n"
        f"  mode {comp_type} \u00b7 stats over\n"
        f"  {layer_km[0]:.1f}\u2013{layer_km[1]:.1f} km band"
    )
    ax_info.text(0.0, 1.0, info, va='top', ha='left', fontsize=10,
                 family='monospace', linespacing=1.5)

    # =======================================================================
    # SCENE  (rows 2-3, cols 0-2) : single ATL-TC classification curtain
    # quicklook_ATC accepts a single axis (parent passes axes=ax5).
    # =======================================================================
    ax_tc = fig.add_subplot(gs[2:4, 0:3])
    ecplt.quicklook_ATC(atc_100km, hmax=hmax[0], resolution=resolution,
                        dstdir=None, axes=ax_tc, comparison=True,
                        station=shortest_time, show_temperature=True)
    ax_tc.set_title('What was the scene \u2014 ATL-TC target classification',
                    fontsize=12, loc='left')
    _z0, _z1 = ax_tc.get_ylim()
    _sc = 1000.0 if max(abs(_z0), abs(_z1)) > 100.0 else 1.0
    _lo, _hi = layer_km[0] * _sc, layer_km[1] * _sc
    if _hi > min(_z0, _z1) and _lo < max(_z0, _z1):
        ax_tc.axhspan(_lo, _hi, fill=False, ec='#1f77b4', lw=1.4,
                      ls=(0, (4, 2)), zorder=5)

    # =======================================================================
    # PROFILES  (cols 4,5,6,7)  -- EXACT parent loop, one var per column
    # =======================================================================
    ax8 = fig.add_subplot(gs[0:4, 4])
    ax9 = fig.add_subplot(gs[0:4, 5])
    ax10 = fig.add_subplot(gs[0:4, 6])
    ax11 = fig.add_subplot(gs[0:4, 7])
    axes = [ax8, ax9, ax10, ax11]

    variables = [
        'particle_backscatter_coefficient_355nm',
        'particle_extinction_coefficient_355nm',
        'lidar_ratio_355nm',
        'particle_linear_depol_ratio_355nm',
    ]
    titles = ['Bsc. Coef.', 'Ext. Coef.', 'Lidar Ratio', 'Lin. depol. ratio']

    xlims = DEFAULT_CONFIG_L2['DEFAULT_XLIMS'] if lin_scale else None
    xlims_log = DEFAULT_CONFIG_L2['DEFAULT_XLIMS_LOG'] if log_scale else None

    # parent's exact idx handling (idxx already captured above)
    if comp_type in ['average', 'average_profiles']:
        idx = slice(None)
    else:
        idx = idx

    # parent's exact profile loop (verbatim)
    for i, (variable, ax, title) in enumerate(zip(variables, axes, titles)):
        if variable in gnd_profiles:
            plot_AEBD_profiles(gnd_profiles, variable, ax=ax, lin_scale=lin_scale,
                               hmax=8e3, log_scale=log_scale, profile='GND',
                               yticks=(i == 0), smoothing=smoothing)
        plot_AEBD_profiles(aebd_profiles, variable, hmax=hmax[2], resolution=resolution,
                           ax=ax, lin_scale=lin_scale, idx=idx,
                           log_scale=log_scale, title=title, profile='EC',
                           xlim=xlims[i] if xlims else None,
                           xlim_log=xlims_log[i] if xlims_log else None,
                           yticks=(i == 0))

    # comparison-layer band, drawn AFTER plotting in each axis's own y-units
    for ax in axes:
        z0, z1 = ax.get_ylim()
        scale = 1000.0 if max(abs(z0), abs(z1)) > 100.0 else 1.0
        lo, hi = layer_km[0] * scale, layer_km[1] * scale
        if hi > min(z0, z1) and lo < max(z0, z1):
            ax.axhspan(lo, hi, color='#A9CCEE', alpha=0.25, lw=0, zorder=0)

    # =======================================================================
    # NEW PANEL 1 : BY HOW MUCH  (rows 0-2, cols 8-9)
    # =======================================================================
    stats = [compute_layer_stats(aebd_profiles, gnd_profiles, v,
                                 hvar='JSG_height', layer_km=layer_km)
             for v in variables]
    short = [r'$\beta$', r'$\alpha$', r'$S$', r'$\delta$']

    ax_bias = fig.add_subplot(gs[0:2, 8])
    rels = [s['rel_bias_pct'] for s in stats]
    ypos = np.arange(len(rels))[::-1]
    finite_rels = [r for r in rels if np.isfinite(r)]
    # adapt the x-range to the data (min +/-30 %, expand if real biases are larger)
    lim = 30.0
    if finite_rels:
        lim = max(30.0, 1.2 * max(abs(r) for r in finite_rels))
    lim = min(lim, 300.0)  # don't let one garbage value blow the axis out
    ax_bias.axvspan(-10, 10, color='#cfe3c4', alpha=0.45, lw=0)
    ax_bias.axvline(0, color='#888', lw=0.9)
    for y, rel in zip(ypos, rels):
        if np.isfinite(rel):
            rc = max(-lim, min(lim, rel))
            ax_bias.plot([0, rc], [y, y], color='#999', lw=1.2, zorder=1)
            ax_bias.scatter([rc], [y], s=90, color='#1f77b4', zorder=3)
            if abs(rel) > lim:  # mark clipped values
                ax_bias.annotate(f'{rel:+.0f}%', (rc, y),
                                 textcoords='offset points', xytext=(-4, 6),
                                 fontsize=7, color='#a33', ha='right')
        else:
            ax_bias.annotate('n/a', (0, y), textcoords='offset points',
                             xytext=(0, 6), fontsize=7, color='#999', ha='center')
    ax_bias.set_yticks(ypos)
    ax_bias.set_yticklabels(short, fontsize=15)
    ax_bias.set_xlim(-lim, lim)
    ax_bias.set_xlabel('EC \u2212 GND  rel. bias [%]', fontsize=11)
    ax_bias.set_title('By how much', fontsize=14, fontweight='bold', loc='left', pad=16)
    ax_bias.text(0.0, 1.015,
                 f'layer {layer_km[0]:.1f}\u2013{layer_km[1]:.1f} km \u00b7 \u00b110 % shaded',
                 transform=ax_bias.transAxes, fontsize=9, color='#555')
    ax_bias.grid(True, axis='x', lw=0.4, color='#ddd')

    # =======================================================================
    # NEW PANEL 2 : STATS TABLE  (rows 2-4, cols 8-9)
    # =======================================================================
    ax_tab = fig.add_subplot(gs[2:4, 8])
    ax_tab.axis('off')

    def _fmt(v):
        if not np.isfinite(v):
            return '\u2013'
        return f'{v:.2g}' if abs(v) >= 0.01 else f'{v:.1e}'

    rows = []
    for s, sh in zip(stats, short):
        rows.append([sh, _fmt(s['ec_mean']), _fmt(s['gnd_mean']),
                     ('\u2013' if not np.isfinite(s['rel_bias_pct'])
                      else f"{s['rel_bias_pct']:+.0f}"),
                     _fmt(s['rmsd'])])
    tab = ax_tab.table(cellText=rows,
                       colLabels=['Var', 'EC', 'GND', '\u0394%', 'RMSD'],
                       loc='upper center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    tab.scale(1, 1.8)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#ccc')
        if r == 0:
            cell.set_facecolor('#eef3f8')
            cell.set_text_props(weight='bold')

    # ----- shared legend ----------------------------------------------------
    lg = [mlines.Line2D([], [], color='#B68AC9', lw=2, label='Ground'),
          mlines.Line2D([], [], color='#1f77b4', lw=2, label='EarthCARE'),
          Rectangle((0, 0), 1, 1, fc='#A9CCEE', alpha=0.4,
                    label=f'Layer {layer_km[0]:.1f}\u2013{layer_km[1]:.1f} km')]
    fig.legend(handles=lg, loc='upper left', bbox_to_anchor=(0.40, 0.945),
               fontsize=12, frameon=False, ncol=3, columnspacing=1.6)

    # ----- save : copied filename convention -------------------------------
    overpass_time_s = pd.Timestamp(time.item()).strftime('%d_%m_%Y_%H_%M_%S.%f')[:-7]
    tag = {'average': 'avg', 'average_profiles': 'avg_prof'}.get(comp_type, str(idxx))
    if dstdir:
        if network == 'LICHT':
            dstfile = f'{overpass_time_s}_L2_cols_{keyword}.png'
        else:
            dstfile = (f'{network}_{keyword}{gnd_ov_time_fname}_{resolution}_'
                       f'{DEFAULT_CONFIG_L2["MAX_DISTANCE"]}_{tag}_cols_'
                       f'{baseline[0]}.png')
        fig.savefig(os.path.join(dstdir, dstfile), bbox_inches='tight', dpi=300)

    fig.subplots_adjust(top=0.85, bottom=0.07, left=0.045, right=0.985,
                        hspace=0.55, wspace=0.5)
    return fig



def plot_EC_L2_comparison(aebdpath, atcpath, gndfolderpath, dstdir, resolution, 
                          fig_scale, network, max_distance=DEFAULT_CONFIG_L2['MAX_DISTANCE'],
                          hmax=DEFAULT_CONFIG_L2['HMAX'], hmin=DEFAULT_CONFIG_L2['HMIN'],
                          raman=True, klett=False,
                          figsize=DEFAULT_CONFIG_L2['FIGSIZE'], 
                          smoothing = DEFAULT_CONFIG_L2['SMOOTHING'],    
                          comp_type = DEFAULT_CONFIG_L2['COMP_TYPE']):
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
    hmin: float              | Profiles: data below hmin (m) is blanked (def: 0)
    lin_scale: bool          | Use linear scale for profiles (default: True)
    log_scale: bool          | Use logarithmic scale for profiles (default: False)
    figsize: tuple           | Figure size in inches (width, height) (def: (35,20))
    
    Returns
    -------
    fig: matplotlib.figure   | The generated comparison plot figure
    """
    print('Start file loading')
    # Load and process GND data
    gnd_quicklook, gnd_profile, station_name, \
        station_coordinates = load_ground_data(network, gndfolderpath, 'L2',
                                                scc_term= 'b0355')
    
    print('Successfully loaded ground data')
    ###Load and process EarthCARE products####
    # Load and crop AEBD  and  ATC product
    try:
        aebd, aebd_50km, shortest_time, aebd_baseline, distance_idx_nearest, \
            dst_min, s_dist_idx = load_crop_EC_product(
                aebdpath, station_coordinates, product='AEBD',
                max_distance=max_distance, second_trim=False)
    except Exception as e:
        print("AEBD dataset not in range of the station.")
        raise
    
    # Load and crop ATC product
    try:
        atc, atc_100km, atc_baseline = load_crop_EC_product(
            atcpath, station_coordinates, 'ATC', max_distance=max_distance)
    except Exception as e:
        print("ATC dataset not in range of the station.")
        atc, atc_100km, atc_baseline = None, None, None
    
    print('Successfully loaded EarthCARE data')
    
    overpass_date = pd.Timestamp(shortest_time.item()).strftime('%d-%m-%Y %H:%M')
    
    if network == 'POLLYXT' or network == 'EARLINET':
        overpass_date = pd.Timestamp(shortest_time.item())#.strftime('%d-%m-%Y %H:%M')
        #overpass_date = '2024-16-10 12:40:52' #mock value for dummy  files.
        if gnd_quicklook is not None:
            gnd_quicklook = crop_polly_file(gnd_quicklook, overpass_date)
        
    # idx_range  : along-track profiles averaged (shaded in the EC quicklooks)
    # idxx_range : along-track index of each figure produced
    n_track = aebd_50km.sizes['along_track']
    if comp_type == 'average_profiles':
        idx_range = list(range(max(s_dist_idx - 4, 0), min(s_dist_idx + 5, n_track)))
        aebd_profile = aebd_50km.isel(along_track=idx_range).mean('along_track')
        idxx_range = [s_dist_idx]

    elif comp_type == 'profile':
        idxx_range = range(max(s_dist_idx - 2, 0), min(s_dist_idx + 2, n_track))
        idx_range = None                      # single profile -> line marker

    elif comp_type == 'average':
        idx_range = list(range(n_track))
        aebd_profile = aebd_50km.mean('along_track')
        idxx_range = [s_dist_idx]
        
    #pdb.set_trace()
    raman = DEFAULT_CONFIG_L2['RETRIEVAL'] != 'KLETT'  # True unless RETRIEVAL is 'KLETT'
    klett = DEFAULT_CONFIG_L2['RETRIEVAL'] != 'RAMAN'  # True unless RETRIEVAL is 'RAMAN'
    
    baseline = [aebd_baseline, atc_baseline]

    # profiles are indexed (along_track, JSG_height)
    for v in aebd_50km.data_vars:
        if aebd_50km[v].dims == ('JSG_height', 'along_track'):
            aebd_50km[v] = aebd_50km[v].transpose('along_track', 'JSG_height')

    for idx in idxx_range:
        if comp_type == 'profile':
            aebd_profile = aebd_50km.isel(along_track=idx)
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
                                network, keyword, idx_range=idx_range, hmin=hmin, figsize=figsize, smoothing=smoothing,
                                comp_type=comp_type)
        else:
         for time_idx in range(gnd_profile.dims['time']):    
            
            if network == 'LICHT':
                gnd_data = gnd_profile.isel(time=time_idx)
                plot_sub_L2(idx, resolution, gnd_quicklook, station_name,
                           station_coordinates, aebd, aebd_50km,
                           shortest_time, baseline, distance_idx_nearest,
                           dst_min, aebd_profile, atc, atc_100km,
                           gnd_data, dstdir, hmax, fig_scale, 
                           network, idx_range=idx_range, hmin=hmin, figsize=figsize, smoothing=smoothing,
                           comp_type=comp_type)
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
                           network, keyword, idx_range=idx_range, hmin=hmin, figsize=figsize, smoothing=smoothing,
                           comp_type=comp_type)