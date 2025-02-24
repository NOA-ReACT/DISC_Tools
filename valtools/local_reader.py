#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:06:09 2025

Contains the reader functions for Thelisys and RV meteor 


@author: akaripis
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import sys
import datetime
import netCDF4

DEFAULT_CONFIG_R = {'convfactor_dp355': 0.739,
                    'convfactor_dp355_err': 0.1,
                    'max_distance_km': 100,
                    'time_window': 60, #minutes
                    'ovp_time': '15:52:00.00', #UTC
                    'gnd_station': 'RV Meteor' # Thelisys
                     }

# part. depol Conversion factor from 532 to 355 nm (Dedicate -> https://nebula.esa.int/sites/default/files/neb_study/1219/C4000112750ExS.pdf)

def conv_dp355(dt_gnd, dp532_id, convfactor_dp355, convfactor_dp355_err):

    # dp532 : part depol 532 nm -> xr.Dataset values 2D array [time, height/alt]
    # dp532_error : part depol 532 nm error -> xr.Dataset values 2D array [time, height/alt]
    # dp355 : converted part depol 355 nm -> xr.Dataset with Dims [time, height/alt]
    # convfactor_dp355 : scalar value
    # convfactor_dp355_err : scalar value

    # dt_gnd[dp355_id]
    dp355 = dt_gnd[dp532_id].copy() * convfactor_dp355 #units 1 # converted depol 532 to 355 nm
    
    dp532 = dt_gnd[dp532_id].values # depol 532
    dp532_error = dt_gnd[f'{dp532_id}_err'].values # depol 532 error
    
    # error propagation for yi = c*xi -> yi_err = yi*sqrt((c_err/c)^2 + (xi_err/xi)^2)
    # dt_gnd[f'{dp355_id}_err'] 
    dp355_err = dp355 * np.sqrt(np.power(convfactor_dp355_err/convfactor_dp355,2) +\
                                                           np.power(dp532_error/dp532,2)) 
    
    return(dp355, dp355_err)

def extract_ovp_time_excel(excel_file, date_str):
    """Reads the excel file and extracts the overpass time in HH:MM:SS.000000 format"""
    # Read excel file
    excel_dt = pd.read_excel(excel_file[0], header=4)
    
    # Convert dates properly
    excel_dt['OVERPASS_DATE'] = pd.to_datetime(excel_dt['OVERPASS_DATE'])
    
    # Match the date format with input date_str
    date_mask = excel_dt['OVERPASS_DATE'].dt.strftime('%Y-%m-%d') == date_str
    
    # Get the matching row
    masked_dt = excel_dt[date_mask]
    
    try:
        ovp_time = masked_dt.iloc[0]['OVERPASS_TIME_UTC'].strftime("%H:%M:%S.%f")
    except:
        raise ValueError(f'The overpass date {date_str} does not match LICHT excel entry!\nCheck the excel file {excel_file[0]}')
    
    return ovp_time


def read_RV_meteor(data_path, time_window, max_distance_km=100,
                   convfactor_dp355=0.86, convfactor_dp355_err=0.1,
                   station_name='RV Meteor'):
        

    # ground_dir = os.path.join(root_dir, 'L2', 'gnd')
    # folder_path_gnd = os.path.join(ground_dir)

    # Read licht data
    gnd_parser = 'LICHT*v1.nc'
    gnd_file = glob.glob(os.path.join(data_path,gnd_parser))

    # Excel file to get the overpass time info
    # excel_file = glob.glob(os.path.join(os.path.split(data_path)[0],'LICHT*.xlsx'))
    excel_parts=data_path.split('/')
    excel_path = os.path.join(*excel_parts[0:5])
    excel_file =glob.glob(os.path.join((excel_path),'LICHT*.xlsx'))

    dt_gnd = []
    dt_gnd_ovp = []

    if len(gnd_file)>0:
        dt_gnd = xr.open_dataset(gnd_file[0], engine='netcdf4', chunks='auto') # chunks auto to reduce memory usage
        dt_gnd = dt_gnd.where(dt_gnd != netCDF4.default_fillvals['f8'])
        dt_gnd.close()
        
        products_list = list(dt_gnd.keys())
        
        # products to be extracted from b files
        bp355_id = 'bp355'
        dp532_id = 'dp532'
        
        # products to be calculated/created
        ap355_id = 'ap355'
        lr355_id = 'lr355'
        dp355_id = 'dp355'
            
        # Datetimes 
        # date_str = pd.Timestamp(dt_gnd['time'].values[0]).strftime('%Y%m%d')# format YYYYMMDD 
        date_str = pd.Timestamp(dt_gnd['time'].values[0]).strftime('%Y-%m-%d')# format YYYY-MM-DD 
        
        # Extracts the ovp time from excel file based on the date of LICHT data    
        ovp_time = extract_ovp_time_excel(excel_file, date_str)     
        ovp_datetime = np.datetime64(f'{date_str}T{ovp_time}')
        
        # time slice for the window ovp_datetime +- time_window
        time_slice = slice(ovp_datetime-np.timedelta64(time_window,'m'),ovp_datetime+np.timedelta64(time_window,'m'))
        
        # Calculate the slope (diff) to find where the lat and lon do not change with --> fixed location for the overpass
        lat_dif = dt_gnd["lat"].loc[time_slice].diff('time').values
        lon_dif = dt_gnd["lon"].loc[time_slice].diff('time').values
        # dt_gnd['time'].loc[time_slice].where((lon_dif == 0))
        
        # Mask to find the position where lat and lon do not change with time 
        mask_fix_loc = (lat_dif == 0.) & (lon_dif == 0.) #(lat_dif < 1e-6) & (lon_dif < 1e-6) #
        mask_fix_loc = np.insert(mask_fix_loc,0,False) # make the size (lat_diff) equal to original array (lat)
        
        #dt_gnd["lat"].loc[time_slice].where(mask_fix_loc)  # dt_gnd["time"].loc[time_slice][1:][mask_fix_loc]
        
        # Select only part of the lidar dataset beased on the time window (ovp +- time_window)
        dt_gnd = dt_gnd.sel(time=time_slice)
        
        # Extract the lat and lon values of Meteor's fixed location for EC overpass
        gnd_lat = dt_gnd["lat"].where(mask_fix_loc).mean(skipna=True).values #.loc[time_slice]
        gnd_lon = dt_gnd["lon"].where(mask_fix_loc).mean(skipna=True).values #.loc[time_slice]
        gnd_coordinates = [gnd_lat.item(), gnd_lon.item()]
        
        # gnd_altitude = dt_gnd["alt"].values # units m # height above mean sea level
        
        time_gnd = dt_gnd["time"].values
        
        # Licht indexes only during fixed location - to be used for comparison 
        ovp_idxs = time_gnd[mask_fix_loc]
        
        # beta
        dt_gnd[bp355_id].values = dt_gnd[bp355_id].values
        dt_gnd[bp355_id].attrs['units'] = '1/m/sr'
        
        dt_gnd[f'{bp355_id}_err'].values= dt_gnd[f'{bp355_id}_err'].values
        dt_gnd[f'{bp355_id}_err'].attrs['units'] = '1/m/sr'
        
        # alpha     
        dt_gnd[ap355_id]= dt_gnd[bp355_id].copy() * np.nan
        dt_gnd[f'{ap355_id}_err']= dt_gnd[f'{bp355_id}_err'].copy() * np.nan
        dt_gnd[ap355_id].attrs['units'] = dt_gnd[f'{ap355_id}_err'].attrs['units'] = '1/Mm'
        
        # lidar ratio
        dt_gnd[lr355_id]= dt_gnd[bp355_id].copy() * np.nan
        dt_gnd[f'{lr355_id}_err']= dt_gnd[f'{bp355_id}_err'].copy() * np.nan
    
        #par depol 355 (converted from 532)
        dt_gnd[dp355_id], dt_gnd[f'{dp355_id}_err'] = conv_dp355(dt_gnd, dp532_id, 
                                                                 convfactor_dp355, convfactor_dp355_err)
    
        # Select only the ids for comparison and rename for plotting routines
        rename_ids = dict(alt='height', bp355='particle_backscatter_coefficient_355nm', bp355_err='particle_backscatter_coefficient_355nm_error',
                          dp355 = 'particle_linear_depol_ratio_355nm', dp355_err = 'particle_linear_depol_ratio_355nm_error',
                          lr355='lidar_ratio_355nm', lr355_err='lidar_ratio_355nm_error', 
                          ap355='particle_extinction_coefficient_355nm', ap355_err ='particle_extinction_coefficient_355nm_error',
                          lat='latitude', lon='longitude')
    
        #dt_gnd[prods_id[0]].loc[dict(time=time_slice)][mask_fix_loc,:].plot(hue='time')
        
        # Dataset with profiles within the given time window from ovp time
        dt_gnd = dt_gnd[rename_ids.keys()].rename(rename_ids)
        
        # Dataset with profiles only during the fixed location (EC ovp)
        dt_gnd_ovp = dt_gnd.sel(time=ovp_idxs)

        return dt_gnd, dt_gnd_ovp, gnd_coordinates, station_name
    
    else:
        print('No file found')


# ship_tot, ship_cropped, a, b = read_RV_meteor(root_dir='/home/akaripis/earthcare/files/20240820', 
#                                          oveprass_time='15:52:00.00', time_window=60)
