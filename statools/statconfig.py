#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for EarthCARE-Ground validation statistical analysis.

This module contains all configuration parameters, constants, and settings used
throughout the statistics workflow. 

The main configuration dictionary DEFAULT_CONFIG_ST controls:
- Event selection and data paths
- Processing parameters (distance thresholds, height limits)
- Analysis methods (retrieval types, grouping methods)
- Filtering and outlier removal settings
- Plotting and visualization parameters

Created on Sun Jul 20, 2025

@author: Andreas Karipis
@affiliation: National Observatory of Athens (NOA), ReACT
@contact: akaripis@noa.gr
@version: 1.0
"""

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

DEFAULT_CONFIG_ST = {
    # Event Selection
    # ---------------
    # List of event directories containing EarthCARE and ground-based data
    # Each path should point to a directory with AEBD and ground station files
    'EVENT_LIST': [
        '/home/akaripis/earthcare/files/20250709',
        '/home/akaripis/earthcare/files/20250725',
        '/home/akaripis/earthcare/files/20250803',
    ],
    
    # Alternative event lists for different analysis periods
    # Uncomment and modify as needed:
    # 'EVENT_LIST': [
    #     '/home/akaripis/earthcare/files/20241007',
    #     '/home/akaripis/earthcare/files/20241016', 
    #     '/home/akaripis/earthcare/files/20241212',
    #     '/home/akaripis/earthcare/files/20250306',
    #     '/home/akaripis/earthcare/files/20250315',
    #     '/home/akaripis/earthcare/files/20250322',
    #     '/home/akaripis/earthcare/files/20250416',
    #     '/home/akaripis/earthcare/files/20250425',
    #     '/home/akaripis/earthcare/files/20250511',
    #     '/home/akaripis/earthcare/files/20250520',
    #     '/home/akaripis/earthcare/files/20250605',
    #     '/home/akaripis/earthcare/files/20250614',
    #     '/home/akaripis/earthcare/files/20250630'
    # ],
    
    # Spatial and Temporal Matching
    'MAX_DISTANCE': 50,  # Maximum distance between satellite and ground station (km)
    'NETWORK': 'POLLYXT',  # Ground-based lidar network identifier
    
    # Height Range Settings
    'HMAX_KM_BINS': [20, 20, 20, 20],  # Height limits when using km_bins grouping
    'HMAX_HEIGHT': [5, 5, 5, 5],       # Height limits when using height grouping
    
    # Variables of Interest
    'VARIABLES': [
        'particle_backscatter_coefficient_355nm',  # Aerosol backscatter at 355nm
        'particle_extinction_coefficient_355nm',   # Aerosol extinction at 355nm  
        'lidar_ratio_355nm',                       # Extinction-to-backscatter ratio
        'particle_linear_depol_ratio_355nm'        # Linear depolarization ratio
    ],
    
    # Processing Options
    'RETRIEVAL': 'raman',        # Retrieval method: 'raman', 'klett', or 'both'
    'RESOLUTION': 'low',         # Data resolution: 'low' or 'high'
    'CERTAIN_PROFILES': True,    # Use only closest satellite profiles (True/False)
    'SMOOTH_GROUND_DATA': False, # Apply Savitzky-Golay smoothing to ground data
    
    # Statistical Analysis
    'GROUPING_METHOD': 'height',  # Grouping method: 'height' or 'km_bins'
    
    # Cloud Filtering
    'ENABLE_FILTERING': True,              # Enable cloud filtering
    'FILTER_VARIABLE': 'simple_classification',  # Variable used for filtering
    'FILTER_VALUES': [1, 2, 4],          # Classification values to filter out
    
    # Outlier Detection and Removal
    'REMOVE_OUTLIERS': True,    # Enable outlier removal for scatter plots
    'OUTLIER_METHOD': 'iqr',    # Method: 'iqr', 'zscore', or 'percentile'
    'OUTLIER_FACTOR': 1.5,      # Threshold factor (1.5=mild, 3.0=extreme outliers)
    
    # Plotting and Visualization
    'FIGSIZE': (35, 20),      # Default figure size (width, height) in inches
    'FIG_SCALE': 'linear',    # Figure scale: 'linear' or 'log'
    'PLOT_BOTH': False,       # Plot both filtered and unfiltered data
}

# =============================================================================
# VARIABLE MAPPING AND CONSTANTS
# =============================================================================

# Standard atmospheric variable names and their corresponding units
VARIABLE_UNITS = {
    'particle_backscatter_coefficient_355nm': 'm^-1 sr^-1',
    'particle_extinction_coefficient_355nm': 'm^-1', 
    'lidar_ratio_355nm': 'sr',
    'particle_linear_depol_ratio_355nm': 'dimensionless'
}

# Scaling factors for different variables (for visualization)
SCALING_FACTORS = {
    'particle_backscatter_coefficient_355nm': 1e6,  # Convert to Mm^-1 sr^-1
    'particle_extinction_coefficient_355nm': 1e6,   # Convert to Mm^-1
    'lidar_ratio_355nm': 1,                         # No scaling
    'particle_linear_depol_ratio_355nm': 1          # No scaling
}

# Default axis limits for scatter plots (after scaling)
AXIS_LIMITS = {
    'particle_backscatter_coefficient_355nm': (-1, 6),
    'particle_extinction_coefficient_355nm': (-20, 150),
    'lidar_ratio_355nm': (-20, 200),
    'particle_linear_depol_ratio_355nm': (-0.1, 0.6)
}