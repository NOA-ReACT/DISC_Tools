#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:53:10 2024

EarthCARE L2 ATLID Data Visualization Tool.

This tool creates comparison plots between EarthCARE L2 ATLID data (AEBD, ATC) 
and ground-based measurements from lidar stations.

Visualization Components:
    1. Five quicklooks of L2 products
    2. Two quicklooks from ground station
    3. Four profile comparisons
    4. Classification and quality status plot
    5. Geographical map

Author: Andreas Karipis
Institution: Remote Sensing of Aerosols, Clouds and Trace gases (ReaCT)
Organization: National Observatory of Athens (NOA)
Version: 1.0.0
"""

import sys
sys.path.append('/home/akaripis/earthcare')
import matplotlib.pyplot as plt

from valtool_manager import plot_EC_L2_comparison, DEFAULT_CONFIG_L2
from valio import build_paths




def main():
    """
    Main execution function for the EarthCARE L2 visualization tool.
    
    Parameters:
    -----------
    aebdpath (str):             | Path to ANOM data file
    atcpath (str):              | Path to simulator data file
    sccfolderpath (str):        | Path to SCC data folder
    pollypath (tr) :            | Path to PollyXT data foler
    distdir (str):              | Output directory for plots
    resolution (str):           | Resolution of AEBD products to be plotted. 
                                    Possible values: High, med, low. 
    lin_scale (Bool):           | Whether to use linear scale for profile plots, 
                                    default: True
    log_scale (Bool):           | Whether to use logarithmic scale for profile 
                                    plots, default: False
    scale: str                  | Scale of the profiles: linear or log
    """
    # Input paths

    root_dir = "/home/akaripis/earthcare/files/20241212"
    PATHS = build_paths(root_dir, 'L2')
    
    try:
        fig = plot_EC_L2_comparison(aebdpath=PATHS['AEBD'], atcpath=PATHS['ATC'],
                                  sccfolderpath=PATHS['SCC'], pollypath=PATHS['POLLY'],
                                  dstdir=PATHS['OUTPUT'], resolution='high',
                                  network = DEFAULT_CONFIG_L2['NETWORK'], 
                                  fig_scale =DEFAULT_CONFIG_L2['FIG_SCALE'])
        plt.show()
    except Exception as e:
        print(f'Error in main execution: {str(e)}')
        raise


if __name__ == '__main__':
    main()
