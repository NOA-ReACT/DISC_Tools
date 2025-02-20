#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:53:10 2024

EarthCARE L1 ATLID Data Visualization Tool.

This tool creates comparison plots between EarthCARE L1 ATLID data, simulator 
data, and ground station measurements. It generates a comprehensive visualization 
layout combining quicklooks, profiles, and geographical context.

Visualization Components:
    1. Three quicklooks of the L1 A-NOM product:
       - Mie attenuated backscatter
       - Rayleigh attenuated backscatter
       - Cross-polar attenuated backscatter
    2. Two quicklooks from the ground station:
       - Range corrected signal
       - Volume depolarization ratio
    3. Three profiles comparing the same variables
    4. Map showing:
       - Station location
       - Orbit track (black)
       - Data acquisition track (red)
       - 100km radius around station

Author: Andreas Karipis
Institution: Remote Sensing of Aerosols, Clouds and Trace gases (ReaCT)
Organization: National Observatory of Athens (NOA)
Version: 1.0.0
"""
import sys
sys.path.append('/home/akaripis/earthcare')
import matplotlib.pyplot as plt

from valtool_manager import plot_EC_L1_comparison
from valio import build_paths



def main():
    """
    Main execution function for the EarthCARE L1 visualization tool.
    
    Parameters:
    -----------
    anompath (str):                 | Path to ANOM data file
    simpath (str):                  | Path to simulator data file
    sccfolderpath (str):            | Path to SCC data folder
    distdir (str):                  | Output directory for plots
    lin_scale (Bool):               | Whether to use linear scale for profile 
                                        plots, default: True
    log_scale (Bool):               | Whether to use logarithmic scale for 
                                        profile plots, default: False
    """
    # Input paths
    root_dir = "/home/akaripis/earthcare/files/20241212"
    PATHS = build_paths(root_dir, 'L1')
    
    try:
        fig = plot_EC_L1_comparison(anompath=PATHS['ANOM'], simpath=PATHS['SIM'],
                                   sccfolderpath=PATHS['SCC'], pollyforlderpath=PATHS['POLLY'],
                                   dstdir=PATHS['OUTPUT'],network='EARLINET',lin_scale=False, 
                                   log_scale=True)

        plt.show()
    except Exception as e:
        print(f'Error in main execution: {str(e)}')
        raise


if __name__ == '__main__':
    main()