#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
valtools - EarthCARE ATLID Validation Tools

A subpackage of ectools for creating comparison plots between EarthCARE 
ATLID data and ground-based lidar network measurements (EARLINET, POLLYXT, 
LICHT, THELISYS).

Usage
-----
    from ectools.valtools import plot_EC_L1_comparison, plot_EC_L2_comparison

Authors: Andreas Karipis, Maria Tsichla, Peristera Paschou, Eleni Marinou, Ping Wang
Contact: a.karipis@noa.gr, elmarinou@noa.gr
Version: 2.0.0
"""

from .valtool_manager import plot_EC_L1_comparison, plot_EC_L2_comparison
from.valio import load_crop_EC_product
from .valplot import plot_AEBD_profiles,plot_AEBD_cla_qs, plot_ANOM_profiles
__all__ = [
    'plot_EC_L1_comparison',
    'plot_EC_L2_comparison',
    'load_crop_EC_product',
    'plot_AEBD_profiles',
    'plot_AEBD_cla_qs',
    'plot_ANOM_profiles'
]

__version__ = '2.0.0'
__author__  = 'Andreas Karipis','Eleni Marinou'
__contact__ = 'a.karipis@noa.gr','elmarinou@noa.gr'