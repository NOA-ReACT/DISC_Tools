#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:06:54 2025

@author: akaripis
"""

##############################################################################
#DICTIONARIES
#############################################################################


classification_color_dict = {-3: 'pink',
                             -2: 'deeppink',
                             -1: 'violet',
                              0: 'blueviolet',
                              1: 'blue',
                              2: 'cornflowerblue',
                              3: 'cyan',
                              5: 'magenta',
                             10: 'teal',
                             11: 'aquamarine',
                             12: 'lime',
                             13: 'green',
                             14: 'olivedrab',
                             15: 'yellow',
                             20: 'palegoldenrod',
                             21: 'gold',
                             22: 'darkgoldenrod',
                             25: 'darkorange',
                             26: 'bisque',
                             27: 'lightsalmon',
                             101: 'orangered',
                             102: 'red',
                             104: 'darkred',
                             105: 'lightgray',
                             106: 'grey',
                             107: 'black'}


classification_dict = {-3:'Missing Data',
                       -2:'(sub-) Surface', #'(sub-) Surface or sub-surface'
                       -1:'Noise', # in both Mie and Ray Channels',
                        0:'Clear',
                        1:'Liquid Cloud', # '(Warm) Liquid Cloud'
                        2:'(Supercooled) \n Liquid Cloud', # (Supercooled) Liquid Cloud
                        3:'Ice Cloud',
                        5: 'Uknown',
                       10:'Dust',
                       11:'Sea salt',
                       12:'Pollution', # 'Continental Pollution'
                       13:'Smoke',
                       14:'Dusty smoke',
                       15:'Dusty mix',
                       20:'STS',
                       21:'NAT',
                       22:'Strat. Ice', # 'Stratospheric Ice'
                       25:'Strat. Ash', #'Stratospheric_Ash'
                       26:'Strat. Sulfate', # 'Stratospheric_Sulfate'
                       27:'Strat. Smoke', # 'Stratospheric_Smoke'
                      101:'Unknown AT',#: Aerosol Target has a very low probability (no class assigned)',
                      102:'Unknown AC',#: Aerosol classification outside of param space',
                      104:'Unknown SAT',#: Strat. Aerosol Target has a very low probability (no class assigned)',
                      105:'Unknown SAC',#: Strat. Aerosol classification outside of param space',
                      106:'Unknown PSCT',#: PSC Target has a very low probability (no class assigned)',
                      107:'Unknown PSCC',}#: PSC classification outside of param space'}


aerosol_classification_color_dict = {1: 'deeppink',
                                     2: 'blue',
                                     3: 'green',
                                     4: 'yellow',
                                     5: 'orange',
                                     6: 'red',
                                     7: 'purple',
                                    -1: 'grey',
                                    -2: 'black',
                                     0: 'cyan'}

aerosol_classification_dict = {1: 'Dust',
                               2: 'Sea_salt',
                               3: 'Continental_Pollution',
                               4: 'Smoke',
                               5: 'Dusty_smoke',
                               6: 'Dusty_mix',
                               7: 'Ice',
                              -1: 'Aerosol target has a very low probability (no class)',
                              -2: 'Aerosol classification outside of parameter space',
                               0:'No input from EarthCARE'}



quality_status_color_dict = {0: 'limegreen',
                             1: 'cornflowerblue',
                             2: 'yellow',
                             3: 'orange',
                             4: 'red'}

quality_status_dict = {0: 'Good: Nominal',
                       1: 'Likely Good, but possibly \n degraded or low SNR',
                       2: 'Likely Bad',
                       3: 'Bad or \n unusable retrieval',# (e.g. lidar signals effectively completely attenuated)',
                       4: 'Missing or bad L1 data'}
