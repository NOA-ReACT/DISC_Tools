#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File containing the configurations for the L1_tool.py & L2_tool.py. 

User defines the ROOT_DIR to the data directory as desrcibed in the README file. 
Modifies each of the following parameters according to the case of interest. 

- MAX_DISTANCE: the distance to which the EC data will be cropped around the station. 
- HMAX: the maximum height for the plots.First height entry is for the EC quicklooks,
        second for the GND quicklooks and third for the profiles
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
  
    """

DEFAULT_CONFIG_L1 = {
    'ROOT_DIR': '/home/akaripis/earthcare/files/20250416',
    'BASELINE': 'A*',
    'MAX_DISTANCE': 35,
    'HMAX': [12e3,12e3,12e3],
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
    'ROOT_DIR': '/home/akaripis/earthcare/files/20250306',
    'BASELINE': 'A*',
    'MAX_DISTANCE': 100,
    'HMAX': [20e3,20e3,20e3],
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
    'DEFAULT_XLIMS': [(-1, 10.), (-20, 250), (-20, 200), (-0.1, 0.75)],
    'DEFAULT_XLIMS_LOG': [(5e-2, 5e1), (5e-2, 5e2), (1e1, 2e2), (1e-2, 1e0)],
    # 'DEFAULT_XLIMS_LOG': [(5e-2, 5e1), (5e-1, 5e2), (1e1, 2e2), (1e-2, 1e0)],
    'SMOOTHING': True,
    'COMP_TYPE': 'average', 
    'RETRIEVAL': 'RAMAN' #All caps
}
