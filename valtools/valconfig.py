#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File containing the configurations for the L1_tool.py & L2_tool.py. 

User defines the ROOT_DIR to the data directory as desrcibed in the README file. 
Modifies each of the following parameters according to the case of interest. 

- MAX_DISTANCE: the distance to which the EC data will be cropped around the station. 
- HMAX: the maximum height for the plots.First height entry is for the EC quicklooks,
        second for the GND quicklooks and third for the profiles
- HMIN: L2 profiles only. Data below this height (m) is blanked (not drawn);
        the y-axis still starts at 0. 0 = nothing blanked.
- FIGSIZE: size of the figure. Both L1 & L2 tools are custom made to this size.
           A change to it will alter the whole figure. 
- FIG_SCALE: The scale for the profiles to be plotted.           
- DEFAULT_XLIMS: ax limits for the linear scale profile plots.
- DEFAULT_XLIMS_LOG: ax limits for the log scale profile plots
- NETWORK: Network from which the ground data originate. 
- VARIABLES: variables to be plotted in the profile plots
- RESOLUTION: resolution of EBD data, values: high, medium, low
- BASIC_SMOOTHING: Applies a low pass filter to the ground data to remove high
                    frequency data
- RETRIEVAL: Either RAMAN, KLETT or both
- QS_VAR: A-EBD quality variable used for filtering: 'quality_status' or
          'extended_data_quality_status'
- QS_FILTER: QS_VAR values masked before comparison. None -> no filtering
             quality_status: [2, 3, 4] | extended_data_quality_status: [2, 102, 200]
- QS_BAD_FRACTION: whole A-EBD profile removed if more than this fraction of its
             bins (below HMAX[2]) is bad. 1.0 -> only pixel masking
  
    """

DEFAULT_CONFIG_L1 = {
    'ROOT_DIR': '/home/akaripis/earthcare/files/20250614',
    'BASELINE': 'A*',
    'MAX_DISTANCE': 50,
    'HMAX': [10e3,10e3,10e3],
    'FIG_SCALE': 'log',
    'NETWORK': 'POLLYXT',
    'FIGSIZE': (27, 15),
    'VARIABLES':[
        'mie_attenuated_backscatter',
        'rayleigh_attenuated_backscatter',
        'crosspolar_attenuated_backscatter'
    ],
    'DEFAULT_XLIMS': [(-1, 5), (-0.5, 8), (-2, 5)],
    'DEFAULT_XLIMS_LOG': [(1e-2, 1e1), (1e-1, 1e1), (1e-3, 1e0)], 
}

DEFAULT_CONFIG_L2 = {
    'ROOT_DIR': '/home/akaripis/earthcare/files/20250416/',
    'BASELINE': 'BA',
    'MAX_DISTANCE': 50,
    'HMAX': [10e3,10e3,10e3],
    'HMIN': 0,
    'FIG_SCALE': 'linear',
    'NETWORK': 'POLLYXT',#EARLINET, #THELISYS
    'FIGSIZE': (35, 20),
    'VARIABLES': [
        'particle_backscatter_coefficient_355nm',
        'particle_extinction_coefficient_355nm',
        'lidar_ratio_355nm',
        'particle_linear_depol_ratio_355nm'
    ],
    'RESOLUTION': 'low',
    'DEFAULT_XLIMS': [(-1, 10.), (-20, 150), (-20, 200), (-0.1, 1)],
    'DEFAULT_XLIMS_LOG': [(5e-2, 5e1), (5e-2, 5e2), (1e1, 2e2), (1e-2, 1e0)],
    # 'DEFAULT_XLIMS_LOG': [(5e-2, 5e1), (5e-1, 5e2), (1e1, 2e2), (1e-2, 1e0)],
    'SMOOTHING': True,
    'COMP_TYPE': 'average', #οptions: average(50km), average_profiles(10profiles), profile
    'RETRIEVAL': 'RAMAN', #All caps
    'QS_VAR': 'quality_status',  # or 'extended_data_quality_status'
    'QS_FILTER': [2, 3, 4],           # e.g. [2, 3, 4] | extended: [2, 102, 200]
    'QS_BAD_FRACTION': 1.0,
}

##### - MAAP usage

CUSTOM_PATHS_L2 = {
    'AEBD':   None,  # e.g. h5_url from MAAP
    'ATC':    None,  # e.g. 's3://bucket/path/to/ATL_TC__.h5'
    'GND':    None,  # e.g. '/local/path/to/gnd/folder'
    'OUTPUT': None,  # e.g. './plots'
}

CUSTOM_PATHS_L1 = {
    'ANOM':   None,
    'SIM':    None,
    'GND':    None,
    'OUTPUT': None,
}