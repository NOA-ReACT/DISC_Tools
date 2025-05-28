#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing Module

Processes two types of local data inputs:
1. RV meteor data from the campaign
2. THELISYS data for visualization in the L1/L2 visuaization tool

Author: Andreas Karipis, Peristera Paschou

"""

import xarray as xr
import numpy as np
import pandas as pd
import os
import glob

import netCDF4

DEFAULT_CONFIG_R = {'convfactor_dp355': 1,
                    'convfactor_dp355_err': 0,
                    'max_distance_km': 100,
                    'time_window': 60, #minutes
                    'ovp_time': '15:52:00.00', #UTC
                    'gnd_station': 'RV Meteor' # Thelisys
                     }

# part. depol Conversion factor from 532 to 355 nm (Dedicate -> https://nebula.esa.int/sites/default/files/neb_study/1219/C4000112750ExS.pdf)

def error_propagation_multi(yi, xi, xi_err, c, c_err):
    """
    Calculate error propagation for multiplicative relationships.
    Εrror propagation for yi = c*xi -> 
     yi_err = yi*sqrt((c_err/c)^2 + (xi_err/xi)^2)
    """  
    yi_err = yi * np.sqrt(np.power(c_err/c,2)+np.power(xi_err/xi,2))
    
    return(yi_err)

def conv_dp355(dt_gnd, dp532_id, convfactor_dp355, convfactor_dp355_err):
    """
    Convert particle depolarization from 532nm to 355nm.
    
    Parameters:
    ----------
    dt_gnd: xarray.Dataset | Ground station dataset
    dp532_id: str | Particle depolarization 532nm variable name
    convfactor_dp355: float | Conversion factor for 532nm to 355nm
    convfactor_dp355_err: float | Error of conversion factor
    
    Returns:
    --------
    tuple | (dp355, dp355_err) Converted depolarization at 355nm and its error
    """
    # dp532 : xr.Dataset      | Part depol 532 nm. Values 2D array [time, height/alt]
    # dp532_error : xr.Dataset | Part depol 532 nm error. Values 2D array [time, height/alt]
    # dp355 : xr.Dataset         |converted part depol 355 nm . Dataset dims [time, height/alt]
    # convfactor_dp355 : scalar value
    # convfactor_dp355_err : scalar value
    # dt_gnd[dp355_id]
    dp355 = dt_gnd[dp532_id].copy() * convfactor_dp355 #units 1 # converted depol 532 to 355 nm
    
    dp532 = dt_gnd[dp532_id].values # depol 532
    dp532_error = dt_gnd[f'{dp532_id}_err'].values # depol 532 error
    
    # error propagation 
    dp355_err = error_propagation_multi(dp355, dp532, dp532_error, convfactor_dp355, convfactor_dp355_err)
    
    return(dp355, dp355_err)

def extract_ovp_time_excel(excel_file, date_str):
    """
    Read the excel file and extract the overpass time.
    
    Parameters:
    ----------
    excel_file: list        | List containing the path to the excel file
    date_str: str           | Date string in format 'YYYY-MM-DD'
    
    Returns:
    --------
        str | Overpass time in format 'HH:MM:SS.000000'
    """
    # Read excel file
    excel_dt = pd.read_excel(excel_file[0], header=4)
    
    # Convert dates properly
    excel_dt['OVERPASS_DATE'] = pd.to_datetime(excel_dt['OVERPASS_DATE'])
    
    # Match the date format with input date_str
    date_mask = excel_dt['OVERPASS_DATE'].dt.strftime('%Y-%m-%d') == date_str
    
    # Get the matching row
    masked_dt = excel_dt[date_mask]
    
    try:
        ovp_time = masked_dt['OVERPASS_TIME_UTC'].iloc[0].strftime("%H:%M:%S.%f")
    except:
        raise ValueError(f'The overpass date {date_str} does not match LICHT excel entry!\nCheck the excel file {excel_file[0]}')
    
    return ovp_time


def read_RV_meteor(data_path, time_hwindow, ovp_hwindow=10,
                   convfactor_dp355=DEFAULT_CONFIG_R['convfactor_dp355'], 
                   convfactor_dp355_err=DEFAULT_CONFIG_R['convfactor_dp355_err'],
                   station_name='RV Meteor'):
    """
    Read LICHT data and return processed datasets.
    
    Parameters:
    -----------
    data_path: str              | Path to the LICHT data directory
    time_hwindow: int           | Half-window time in minutes around overpass
    ovp_hwindow: int            | Half-window time in minutes to determine fixed location
    convfactor_dp355: float     | Conversion factor for depolarization from 532nm to 355nm
    convfactor_dp355_err: float | Error of conversion factor
    station_name: str           | Name of the ground station
    
    Returns:
    --------
    tuple | (dt_gnd, dt_gnd_ovp, gnd_coordinates, station_name)
               Ground station dataset, overpass dataset, coordinates, station name
    """


    # Read licht data
    gnd_parser = 'LICHT*v1.nc'
    gnd_file = glob.glob(os.path.join(data_path,gnd_parser))

    excel_path = data_path
    for _ in range(4):
        excel_path = os.path.dirname(excel_path)
    
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

        # Constructs the overpass datetime
        ovp_datetime = np.datetime64(f'{date_str}T{ovp_time}')
        
        # time slice for the window ovp_datetime +- time_window
        time_slice = slice(ovp_datetime-np.timedelta64(time_hwindow,'m'),ovp_datetime+np.timedelta64(time_hwindow,'m'))
        
        # Select only part of the lidar dataset beased on the time window (ovp +- time_window)
        dt_gnd = dt_gnd.sel(time=time_slice)

        # select the time slice (ovp_datetime +- time_window) of licht data 
        time_gnd = dt_gnd["time"].values

        # Calculate the slope (diff) to find where the lat and lon do not change with --> fixed location for the overpass
        lat_dif = dt_gnd["lat"].loc[time_slice].diff('time').values
        lon_dif = dt_gnd["lon"].loc[time_slice].diff('time').values
        # dt_gnd['time'].loc[time_slice].where((lon_dif == 0))
                
        # Mask to find the position where lat and lon do not change with time (assume changes less than 0.0001)
        mask_fix_loc = (lat_dif < 1e-4) & (lon_dif < 1e-4) # (lat_dif == 0.) & (lon_dif == 0.) #(lat_dif < 1e-6) & (lon_dif < 1e-6) #
        mask_fix_loc = np.insert(mask_fix_loc,0,False) # make the size (lat_diff) equal to original array (lat)

        
        #dt_gnd["lat"].loc[time_slice].where(mask_fix_loc)  # dt_gnd["time"].loc[time_slice][1:][mask_fix_loc]
        
        # Mask to select 'ovp_hwindow' minutes (ovp +- ovp_hwindow min) around the overpass to specify fixed location
        mask_fix_time = (time_gnd >= (ovp_datetime-np.timedelta64(ovp_hwindow,'m'))) \
                        & (time_gnd <= (ovp_datetime+np.timedelta64(ovp_hwindow,'m')))
        
        mask_ovp_loc = mask_fix_loc & mask_fix_time
        
        # Extract the lat and lon values of Meteor's fixed location for EC overpass
        gnd_lat = np.round(dt_gnd["lat"].where(mask_ovp_loc).mean(skipna=True).values,decimals=4) #.loc[time_slice]
        gnd_lon = np.round(dt_gnd["lon"].where(mask_ovp_loc).mean(skipna=True).values,decimals=4) #.loc[time_slice]
        gnd_coordinates = [gnd_lat.item(), gnd_lon.item()]
        # gnd_altitude = dt_gnd["alt"].values # units m # height above mean sea level
       
        # Licht indexes only during fixed location - to be used for comparison 
        ovp_idxs = time_gnd[mask_ovp_loc]
        
        # beta
        dt_gnd[bp355_id].values = dt_gnd[bp355_id].values
        # dt_gnd[bp355_id].attrs['units'] = '1/m/sr'
        
        dt_gnd[f'{bp355_id}_err'].values= dt_gnd[f'{bp355_id}_err'].values
        # dt_gnd[f'{bp355_id}_err'].attrs['units'] = '1/m/sr'
        
        # alpha     
        dt_gnd[ap355_id]= dt_gnd[bp355_id].copy() * np.nan
        dt_gnd[f'{ap355_id}_err']= dt_gnd[f'{bp355_id}_err'].copy() * np.nan
        dt_gnd[ap355_id].attrs['units'] = dt_gnd[f'{ap355_id}_err'].attrs['units'] = '1/m'
        
        # lidar ratio
        dt_gnd[lr355_id]= dt_gnd[bp355_id].copy() * np.nan
        dt_gnd[f'{lr355_id}_err']= dt_gnd[f'{bp355_id}_err'].copy() * np.nan
    
        #par depol 355 (converted from 532)
        dt_gnd[dp355_id], dt_gnd[f'{dp355_id}_err'] = conv_dp355(dt_gnd, dp532_id, 
                                                                 convfactor_dp355, convfactor_dp355_err)
    
        # Select only the ids for comparison and rename for plotting routines
        rename_ids = dict(alt='height', bp355='particle_backscatter_coefficient_355nm', bp355_err='particle_backscatter_coefficient_355nm_error',
                          bp532='particle_backscatter_coefficient_532nm', bp532_err='particle_backscatter_coefficient_532nm_error',
                          dp355 = 'particle_linear_depol_ratio_355nm', dp355_err = 'particle_linear_depol_ratio_355nm_error',
                          dv532 = 'volume_linear_depol_ratio_532nm', dv532_err = 'volume_linear_depol_ratio_532nm_error',
                          dp532 = 'particle_linear_depol_ratio_532nm', dp532_err = 'particle_linear_depol_ratio_532nm_error',
                          lr355='lidar_ratio_355nm', lr355_err='lidar_ratio_355nm_error', 
                          ap355='particle_extinction_coefficient_355nm', ap355_err ='particle_extinction_coefficient_355nm_error',
                          lat='latitude', lon = 'longitude')
    
        #dt_gnd[prods_id[0]].loc[dict(time=time_slice)][mask_fix_loc,:].plot(hue='time')
        
        # Dataset with profiles within the given time window from ovp time
        dt_gnd = dt_gnd[rename_ids.keys()].rename(rename_ids)
        
        # Dataset with profiles only during the fixed location (EC ovp)
        dt_gnd_ovp = dt_gnd.sel(time=ovp_idxs[:3])

        return dt_gnd, dt_gnd_ovp, gnd_coordinates, station_name
    
    else:
        raise AttributeError(f'No licht file found in dir: {data_path} \nCheck your input paths')
    

def read_thelisys_data(d, fname, gnd_id):
    """
    Reader function for THELISYS (SULA) profiles that returns an xarray Dataset.
    
    Parameters:
    -----------
    d : netCDF4.Dataset        | The opened netCDF dataset
    fname : str                | Filename of the netCDF file
    gnd_id : str               | Ground identifier
        
    Returns:
    --------
    xr.Dataset
        xarray Dataset containing extracted and processed data
    """
    if gnd_id != 'thelisys':
        return None
    
    # Extract parts of the file name 
    date_str = f'{fname[18:22]}_{fname[22:24]}_{fname[24:26]}'
    time_str = f'{fname[27:31]}_{fname[33:37]}'
    
    # Get altitude and create coordinates
    altitude = d.variables["altitude"][:] * 1E-3  # units km
    
    # Create base xarray dataset with coordinates
    ds = xr.Dataset(
        coords={
            'altitude': ('altitude', altitude),
        },
        attrs={
            'date_str': date_str,
            'time_str': time_str,
            'source': fname,
            'ground_id': gnd_id
        }
    )
    
    # Define variable names and their conversion factors
    var_mapping = {
        'RB355': ('backscatter_raman', 1E6),                # 1/(Mm*sr)
        'RB355_ERROR': ('backscatter_raman_error', 1E6),    # 1/(Mm*sr)
        'KB355': ('backscatter_klett', 1E6),                # 1/(Mm*sr)
        'KB355_ERROR': ('backscatter_klett_error', 1E6),    # 1/(Mm*sr)
        'EXT355': ('extinction', 1E6),                      # 1/Mm
        'EXT355_ERROR': ('extinction_error', 1E6),          # 1/Mm
        'LR355': ('lidar_ratio', 1.0),                      # sr
        'LR355_ERROR': ('lidar_ratio_error', 1.0),          # sr
    }
    
    # Add variables to dataset efficiently
    for nc_var, (xr_var, factor) in var_mapping.items():
        if nc_var in d.variables:
            # Most variables have shape (time=1, altitude)
            data = d.variables[nc_var][0, :] * factor
            ds[xr_var] = ('altitude', data)
            
            # Copy attributes from original dataset if they exist
            if hasattr(d.variables[nc_var], 'attrs'):
                ds[xr_var].attrs = d.variables[nc_var].attrs.copy()
            
            # Add units information based on our conversions
            if '_ERROR' not in nc_var:
                if 'backscatter' in xr_var:
                    ds[xr_var].attrs['units'] = '1/(Mm*sr)'
                elif 'extinction' in xr_var:
                    ds[xr_var].attrs['units'] = '1/Mm'
                elif 'lidar_ratio' in xr_var:
                    ds[xr_var].attrs['units'] = 'sr'
    
    # Handle depolarization separately due to wavelength conversion
    if 'PLDR532' in d.variables:
        # Constants for conversion
        convfactor_dp355 = 0.85
        convfactor_dp355_err = 0.1
        
        # Get original data
        pdr532 = d.variables['PLDR532'][0, :]
        pdr532_err = d.variables['PLDR532_ERROR'][0, :]
        
        # Convert to 355nm wavelength
        pdr355_raman = pdr532 * convfactor_dp355
        pdr355_klett = pdr532 * convfactor_dp355
        
        # Error propagation: yi*sqrt((c_err/c)^2 + (xi_err/xi)^2)
        # Avoid division by zero errors
        valid_indices = (pdr532 != 0)
        pdr355_err = np.zeros_like(pdr532)
        if np.any(valid_indices):
            pdr355_err[valid_indices] = pdr355_raman[valid_indices] * np.sqrt(
                (convfactor_dp355_err/convfactor_dp355)**2 + 
                (pdr532_err[valid_indices]/pdr532[valid_indices])**2
            )
        
        # Add to dataset
        ds['particledepolarization_raman'] = ('altitude', pdr355_raman)
        ds['particledepolarization_raman_error'] = ('altitude', pdr355_err)
        ds['particledepolarization_klett'] = ('altitude', pdr355_klett)
        ds['particledepolarization_klett_error'] = ('altitude', pdr355_err)
        
        # Add metadata
        ds['particledepolarization_raman'].attrs['units'] = 'ratio'
        ds['particledepolarization_raman'].attrs['wavelength'] = '355nm (converted from 532nm)'
        ds['particledepolarization_klett'].attrs['units'] = 'ratio'
        ds['particledepolarization_klett'].attrs['wavelength'] = '355nm (converted from 532nm)'
    
    return ds