#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main orchestration script for EarthCARE-Ground statistical analysis.

Coordinates the complete analysis workflow from data loading through visualization,
combining profile statistics and scatter plot analysis for atmospheric validation.

Main workflow: load_all_events → profile_statistics → create_plots
Dependencies: statconfig, statio, stat_calc, statplot

@author: Andreas Karipis - NOA ReaCT
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import sys
import pdb

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import re

# Local EarthCARE tools
sys.path.append('/home/akaripis/earthcare')
from ectools_noa import ecio, ecplot as ecplt, colormaps as clm
from valtools.valconfig import DEFAULT_CONFIG_L1, DEFAULT_CONFIG_L2
from valtools.valio import *
from valtools.valplot import *

# Local validation modules
from statconfig import DEFAULT_CONFIG_ST
from statio import load_all_events
from stat_calc import profile_statistics
from statplot import create_profile_figure, create_scatter_plots

# =============================================================================
# MAIN ANALYSIS WORKFLOW
# =============================================================================

def main():
    """
    Execute the complete EarthCARE validation analysis workflow.
    
    Parameters
    ----------
    None
        
    Returns
    -------
    None | Executes analysis and saves/displays results
    """
    
    # Get configuration parameters
    event_list = DEFAULT_CONFIG_ST['EVENT_LIST']
    network = DEFAULT_CONFIG_ST['NETWORK']
    max_distance = DEFAULT_CONFIG_ST['MAX_DISTANCE']
    variables = DEFAULT_CONFIG_ST['VARIABLES']
    retrieval = DEFAULT_CONFIG_ST['RETRIEVAL']
    certain_profiles = DEFAULT_CONFIG_ST['CERTAIN_PROFILES']
    
    # Determine retrieval method
    if retrieval == 'klett':
        klett = True
        raman = False
    elif retrieval == 'raman':
        klett = False
        raman = True
    else:
        klett = True
        raman = True
    
    print("="*60)
    print("ATMOSPHERIC PROFILE ANALYSIS WORKFLOW")
    print("="*60)
    
    # A. READ ALL EVENTS AND RETURN DATA LISTS
    print("\nA. LOADING ALL EVENTS...")
    gnd_data_list, sat_data_list, event_names = load_all_events(
        event_list, network, max_distance, klett, raman, 
        certain_profiles, height_min=0, height_max=15e3)
    
    if not gnd_data_list:
        print("No events loaded successfully. Exiting.")
        return
    
    # B. CALCULATE PROFILE STATS
    print("\nB. CALCULATING PROFILE STATISTICS...")
    
    # Determine which stats to calculate based on filtering setting
    if DEFAULT_CONFIG_ST['ENABLE_FILTERING']:
        print("Calculating filtered statistics...")
        plotting_stats = profile_statistics(
            gnd_data_list, sat_data_list, event_names, apply_filtering=True)
        filter_label = "Clouds removed"
    else:
        print("Calculating unfiltered statistics...")
        plotting_stats = profile_statistics(
            gnd_data_list, sat_data_list, event_names, apply_filtering=False)
        filter_label = " "

    # C. PLOT PROFILE STATS
    print("\nC. CREATING PROFILE PLOTS...")
    plot_type = 'mean'
    
    # Generate path variables
    if DEFAULT_CONFIG_ST['CERTAIN_PROFILES']:
        dist_str = '5profiles'
    else:
        dist_str = f'{max_distance}km'
    
    grouping_suffix = f"_{DEFAULT_CONFIG_ST['GROUPING_METHOD']}"
    
    # Create profile plot
    if DEFAULT_CONFIG_ST['ENABLE_FILTERING']:
        filter_suffix = "_clouds_removed" 
        output_path_profile = (f'/home/akaripis/earthcare/files/profile_stats_{plot_type}_'
                              f'{dist_str}{grouping_suffix}{filter_suffix}.png')
    else:
        output_path_profile = (f'/home/akaripis/earthcare/files/profile_stats_{plot_type}_'
                              f'{dist_str}{grouping_suffix}.png')

    print(f"Creating {filter_label.lower()} profile plot...")
    profile_fig = create_profile_figure(
        plotting_stats, 
        None,  # No second dataset
        variables, 
        plot_type, 
        title=f"baseline (Î') - Antikythera overpasses \nOverpass cases: {len(event_names)} - {filter_label}",
        save_path=output_path_profile,
        remove_outliers=False,
        outlier_threshold=5.0)
    
    # D. PLOT SCATTER PLOTS
    print(f"\nD. CREATING SCATTER PLOTS...")

    if DEFAULT_CONFIG_ST['ENABLE_FILTERING']:
        filter_suffix = "_clouds_removed" 
        output_path_scatter = (f'/home/akaripis/earthcare/files/scatter_stats_{plot_type}_'
                              f'{dist_str}{grouping_suffix}{filter_suffix}.png')
    else:
        output_path_scatter = (f'/home/akaripis/earthcare/files/scatter_stats_{plot_type}_'
                              f'{dist_str}{grouping_suffix}.png')

    print(f"Creating {filter_label.lower()} scatter plot...")
    scatter_fig = create_scatter_plots(gnd_data_list, sat_data_list, event_names, 
                                      single_event_idx=None, apply_filtering=True,
                                      variables=None, figsize=(16, 12), show_stats=True, 
                                      save_path=output_path_scatter)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"Events processed: {len(event_names)}")
    print(f"Variables analyzed: {len(variables)}")
    print(f"Grouping method: {DEFAULT_CONFIG_ST['GROUPING_METHOD']}")
    print(f"Filtering enabled: {DEFAULT_CONFIG_ST['ENABLE_FILTERING']}")
    print(f"Plot type: {filter_label}")
    print("="*60)
    
    # Show the figures
    plt.show()

# =============================================================================
# CONFIGURATION SUMMARY AND VALIDATION
# =============================================================================

def print_analysis_summary():
    """
    Print a summary of the current analysis configuration.
    
    Parameters
    ----------
    None
        
    Returns
    -------
    None | Prints configuration summary to console
    """
    print("EarthCARE Validation Analysis Configuration")
    print("=" * 50)
    print(f"Events to process: {len(DEFAULT_CONFIG_ST['EVENT_LIST'])}")
    for i, event in enumerate(DEFAULT_CONFIG_ST['EVENT_LIST'], 1):
        event_name = event.split('/')[-1]
        print(f"  {i}. {event_name}")
    
    print(f"\nAnalysis Parameters:")
    print(f"  Network: {DEFAULT_CONFIG_ST['NETWORK']}")
    print(f"  Max distance: {DEFAULT_CONFIG_ST['MAX_DISTANCE']} km")
    print(f"  Retrieval method: {DEFAULT_CONFIG_ST['RETRIEVAL']}")
    print(f"  Variables: {len(DEFAULT_CONFIG_ST['VARIABLES'])}")
    for var in DEFAULT_CONFIG_ST['VARIABLES']:
        var_short = var.replace('particle_', '').replace('_355nm', '')
        print(f"    - {var_short}")
    
    print(f"\nProcessing Options:")
    print(f"  Grouping method: {DEFAULT_CONFIG_ST['GROUPING_METHOD']}")
    print(f"  Cloud filtering: {DEFAULT_CONFIG_ST['ENABLE_FILTERING']}")
    print(f"  Outlier removal: {DEFAULT_CONFIG_ST['REMOVE_OUTLIERS']}")
    print(f"  Certain profiles: {DEFAULT_CONFIG_ST['CERTAIN_PROFILES']}")
    print(f"  Smooth ground data: {DEFAULT_CONFIG_ST['SMOOTH_GROUND_DATA']}")
    print("=" * 50)


def validate_analysis_setup():
    """
    Validate the analysis configuration before execution.
    
    Parameters
    ----------
    None
        
    Returns
    -------
    bool | True if configuration is valid, False otherwise
    """
    # Check if event list is not empty
    if not DEFAULT_CONFIG_ST['EVENT_LIST']:
        print("ERROR: No events specified in EVENT_LIST")
        return False
    
    # Check height limits consistency
    n_vars = len(DEFAULT_CONFIG_ST['VARIABLES'])
    if len(DEFAULT_CONFIG_ST['HMAX_HEIGHT']) != n_vars:
        print(f"WARNING: HMAX_HEIGHT length ({len(DEFAULT_CONFIG_ST['HMAX_HEIGHT'])}) "
              f"doesn't match number of variables ({n_vars})")
    
    # Check outlier factor validity
    if DEFAULT_CONFIG_ST['OUTLIER_FACTOR'] <= 0:
        print("ERROR: OUTLIER_FACTOR must be positive")
        return False
    
    print("Configuration validation passed.")
    return True

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Print configuration summary
    print_analysis_summary()
    
    # Validate setup
    if validate_analysis_setup():
        # Run main analysis
        main()
    else:
        print("Analysis aborted due to configuration errors.")