#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing module for EarthCARE analysis tools.

"""
import sys
sys.path.append('/home/akaripis/earthcare/valtools')
import os
import glob
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import geopy.distance
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from ectools_noa import ecio
from local_reader import read_RV_meteor#, process_sula_profile
import pdb

def extract_date(filename, keyword, file_type):
    parts = filename.split('_')
        
    try:
        if keyword == 'ECVT' and len(parts) > 7:
            if file_type == 'HIRELPP':
                date_str = parts[7]
            else:
                date_str = parts[6]
            return datetime.strptime(parts[7], '%Y%m%d%H%M')
        
        elif keyword == 'NOA' and len(parts) > 8:
            if file_type == 'profile':
                date_str = parts[0] + parts[1] + parts[2] + parts[8]
            else:
                date_str = parts[0] + parts[1] + parts[2] + parts[5] + parts[6]
            return datetime.strptime(date_str, '%Y%m%d%H%M')
        
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse date from {filename}: {str(e)}")
        return datetime.max
            
    return datetime.max

def build_paths(root_dir, network, level):
    """
    Build paths dictionary from root directory and create directories if they don't exist
    
    Parameters
    ----------
    root_dir : str  | Root directory containing the L1 and L2 structure
    level: str      | Level of data process to search for the equivalent folder: 
                     either L1 or L2
        
    Returns
    -------
    dict
        Dictionary with paths for AEBD, ATC, SCC, POLLY, and OUTPUT
    """
    # Ensure root_dir exists
    if not os.path.exists(root_dir):
        raise ValueError(f"Root directory does not exist: {root_dir}")
    if network == 'EARLINET' or network == 'THELISYS':
        gnd_suffix ='scc'
    elif network == 'POLLYXT':
        gnd_suffix = 'tropos'
    elif network == 'LICHT':
        gnd_suffix = 'licht'
    
    # First create the base directories
    if level =='L1':
        base_dirs = {
            'ANOM': os.path.join(root_dir, level, 'eca'),
            'SIM': os.path.join(root_dir, level, 'sim'),
            'GND': os.path.join(root_dir, level, 'gnd', gnd_suffix),
            'OUTPUT': os.path.join(root_dir, level, 'plots_comparison')
        }
    elif level == 'L2':      
        base_dirs = {
            'AEBD': os.path.join(root_dir, level, 'eca'),
            'ATC': os.path.join(root_dir, level, 'eca'),
            'ACTC': os.path.join(root_dir, level, 'eca'),
            'MRGR': os.path.join(root_dir, level, 'eca'),
            'CFMR': os.path.join(root_dir, level, 'eca'),
            'GND': os.path.join(root_dir, level, 'gnd', gnd_suffix),
            'OUTPUT': os.path.join(root_dir, level, 'plots_comparison')
        }
    
    # Create all base directories
    for key, path in base_dirs.items():
        os.makedirs(path, exist_ok=True)
    
    # Now find files and build the final paths dictionary
    if level =='L1':
        paths = {
            'ANOM': glob.glob(os.path.join(base_dirs['ANOM'], '*ATL_NOM*.h5')),
            'SIM': glob.glob(os.path.join(base_dirs['SIM'], '*ATL_NOM*.h5')),
            'GND': base_dirs['GND'],
            'OUTPUT': base_dirs['OUTPUT']
        }
        paths['ANOM'] = paths['ANOM'][0] if paths['ANOM'] else None
        paths['SIM'] = paths['SIM'][0] if paths['SIM'] else None
    elif level == 'L2':      
        paths = {
            'AEBD': glob.glob(os.path.join(base_dirs['AEBD'], '*ATL_EBD*.h5')),
            'ATC': glob.glob(os.path.join(base_dirs['ATC'], '*ATL_TC__*.h5')),
            'ACTC': glob.glob(os.path.join(base_dirs['ACTC'], '*AC__TC__*')),
            'MRGR': glob.glob(os.path.join(base_dirs['MRGR'], '*MSI_RGR_*')),
            'CFMR': glob.glob(os.path.join(base_dirs['ACTC'], '*CPR_FMR*')),
            'GND': base_dirs['GND'],
            'OUTPUT': base_dirs['OUTPUT']
        }
        paths['AEBD'] = paths['AEBD'][0] if paths['AEBD'] else None
        paths['ATC'] = paths['ATC'][0] if paths['ATC'] else None
        paths['ACTC'] = paths['ACTC'][0] if paths['ACTC'] else None
        paths['MRGR'] = paths['MRGR'][0] if paths['MRGR'] else None
        paths['CFMR'] = paths['CFMR'][0] if paths['CFMR'] else None
    
    return paths


def load_ground_data(network, data_path, data_type='L1', scc_term ='b0355', date=None):
    """
    Load ground-based lidar data based on network type and data requirements.
    
    Parameters:
    -----------
    network : str        | Network type ('EARLINET' or 'POLLYXT')
    dta_path : str       | Path to ground data files
    data_type : str      | Type of data processing required ('L1' or 'L2')
    smoothing: bool      |Apply a low pass filter to remove high frequency noise
    
    Returns:
    --------
    tuple        | (gnd_quicklook, gnd_profile (optional), station_name, station_coordinates)
    """
    if network == 'EARLINET':
        if data_type == 'L1':
            # L1 processing
            gnd_quicklook, station_name, station_coordinates = load_process_scc_L1(data_path)
        else:
            # L2 processing
            if scc_term == 'elda':
                print('elda gnd file')
                try:
                    gnd_profile_b = process_multiple_files(data_path, 'EARLINET', 'elda',date)
                    gnd_profile_e = process_multiple_files(data_path, 'EARLINET', 'e0355', date)
                except Exception:
                    gnd_profile_b = process_multiple_files(data_path, 'EARLINET', 'elda',date)
                    gnd_profile_e = None
                    
            else:
                gnd_profile_b = process_multiple_files(data_path, 'EARLINET', 'b0355',date)
                try:
                    gnd_profile_e = process_multiple_files(data_path, 'EARLINET', 'e0355', date)
                except Exception:
                    gnd_profile_e = None
                print('b0355')
            gnd_profile = ([gnd_profile_b, gnd_profile_e])
            try:
                gnd_quicklook = load_process_scc_L1(data_path,date)
            except Exception:
                gnd_quicklook = None
            station_name = gnd_profile_b.attrs['location'].split(',')[0].strip()
            station_coordinates = [
                    gnd_profile_b['latitude'].values,
                    gnd_profile_b['longitude'].values
                ]
            # if smoothing:
            #     gnd_profile = gnd_profile.map(apply_gaussian_smoothing)
    elif network == 'POLLYXT':
        if data_type == 'L1':
            # L1 processing
            gnd_quicklook_a = process_multiple_files(data_path, 'POLLYXT', 'att_bsc')
            gnd_quicklook_b = process_multiple_files(data_path, 'POLLYXT', 'vol_depol')
            gnd_quicklook = xr.merge([gnd_quicklook_a, gnd_quicklook_b])
            station_name, station_coordinates = get_polly_station_info(gnd_quicklook, data=True)

        else:
            # L2 processing
            try:
                gnd_quicklook = process_multiple_files(data_path, 'POLLYXT', 'quasi_results')
                # The following line needed for the quasi files to be plotted correctly
                # else they are squeezed between 0-3km 
                gnd_quicklook = gnd_quicklook.assign_coords(height=gnd_quicklook.height * 1)
            except Exception:
                gnd_quicklook = None            
            gnd_profile = process_multiple_files(data_path, 'POLLYXT', 'profile')
            station_name, station_coordinates = get_polly_station_info(gnd_profile, data=True)


    elif network == 'LICHT':
        gnd_quicklook, gnd_profile, station_coordinates, station_name = read_RV_meteor(data_path, 
                                                 time_hwindow=60)
    elif network == 'THELISYS':
        gnd_profile, station_coordinates, station_name = read_sula_file(data_path)
        #gnd_quicklook = None
        gnd_quicklook, station_name1, station_coordinates1 = load_process_scc_L1(data_path)
    else:
        raise ValueError(f"Unsupported network: {network}. Must be either 'EARLINET' or 'POLLYXT'")
    

    if data_type == 'L2':
        return gnd_quicklook, gnd_profile, station_name, station_coordinates
    return gnd_quicklook, station_name, station_coordinates
    
def get_polly_station_info(filename, data=False):
    """
    Read NetCDF file and return station name and coordinates.
    
    Parameters:
    -----------
    filename : str | Path to the NetCDF file
    data: bool     | Whether filename is the data file, or the filepath    
        
    Returns:
    --------
    station : str      | station_coordinates
    """
    if data:
        # If filename is already a dataset
        ds = filename
        station = ds.attrs.get('location')
        lat = round(ds['latitude'].item(0), 2)
        lon = round(ds['longitude'].item(0), 2)
        station_coordinates = [lat, lon]

        return station, station_coordinates
    else:
        # If filename is a path to a file
        with xr.open_dataset(filename) as ds:
            station = ds.attrs.get('location')
            lat = round(ds['latitude'].item(0), 2)
            lon = round(ds['longitude'].item(0), 2)
            station_coordinates = [lat, lon]

            return station, station_coordinates

        
def convert_ds_time(ds):
    """
    Convert the time coordinate of quasi dataset from Unix timestamps 
    to datetime64[s] format to match SCC format.
    
    Parameters
    ----------
    quasi_ds : xarray.Dataset   |The quasi dataset with Unix timestamp time coordinates
        
    Returns
    -------
    xarray.Datase               | Dataset with converted time coordinates
    """
    # Convert Unix timestamps to datetime64[ns]
    new_time = pd.to_datetime(ds.time.values, unit='s')
    
    # Create a new dataset with the converted time coordinate
    ds = ds.assign_coords(time=new_time)
    
    # Ensure the time coordinate has the correct encoding
    ds.time.encoding.update({
        'units': 'nanoseconds since 1970-01-01',
        'calendar': 'proleptic_gregorian'
    })
    
    return ds

def crop_polly_file(ds, crop_time, time_window=pd.Timedelta('1.5H')):
    """
    Crop polly file for quicklook around the time of the overpass, default +- 1.5h
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be cropped
    crop_time : datetime-like
        Center time to crop around (e.g., satellite overpass time)
    time_window : pd.Timedelta, optional
        Time window to keep on either side of crop_time. Default is 1.5 hours
        
    Returns
    -------
    xarray.Dataset
        Dataset cropped to the specified time window
    """
    # Convert crop_time to pandas Timestamp if it isn't already
    crop_time = pd.to_datetime(crop_time)
    
    # Calculate time window boundaries
    start_time = crop_time - time_window
    end_time = crop_time + time_window
    
    # Crop the dataset to the time window
    cropped_ds = ds.sel(time=slice(start_time, end_time))

    # Check if we got any data
    if len(cropped_ds.time) == 0:
        raise ValueError(f"No data found in the time window {start_time} to {end_time}")
        
    return cropped_ds
        
def process_multiple_files(folder_path, network, file_type=None, date=None):
    """
    Search for files in a folder matching a keyword, sort them by date,
    and combine into single xarray dataset.
    
    Parameters
    ----------
    folder_path : str      | Path to the folder containing data files
    network : str          | The word that will enable the differrent processing 
                              of the files. Only EARLINET and POLLYNET at the moment.
    file_type:             | For PollyXT files, indication to the filetype: profile, 
                              quasi, att.bsc etc and for ECVT HIRELPP, e0355, b0355
    date:                   | Date of the file to be processed
        
    Returns
    -------
    xarray.Dataset
        Combined dataset
    """
    if network == 'EARLINET':
        keyword = 'ECVT'
    elif network == 'POLLYXT':
        keyword = 'NOA'
    elif network == 'THELISYS':
        keyword ='THELISYS'
    else: 
        raise ValueError('Only EARLINET and POLLYXT networks available for processing')
    # pdb.set_trace()
    all_files = os.listdir(folder_path)
    #pdb.set_trace()
    if network == 'EARLINET':
        files = [f for f in all_files if file_type in f]
    elif network == 'POLLYXT':
        files = [f for f in all_files if file_type in f]
    elif network == 'THELISYS':
        file_type = 'THELISYS'
        files = [f for f in all_files if file_type in f]
    if not files:
        raise ValueError(f'No {network} files found in {folder_path}')
       
    # Apply date filter if provided
    if date is not None:
        date = pd.to_datetime(date)
        date_filtered = []
        for file in files:
                filedate = extract_date(file, keyword, file_type)
                # Ensure filedate is a pandas datetime
                if not isinstance(filedate, pd.Timestamp):
                    filedate = pd.to_datetime(filedate)
                # Fixed the boolean logic with parentheses
                if (filedate > date - pd.Timedelta('2H')) & (filedate < date + pd.Timedelta('2H')):
                    date_filtered.append(file)
        if not date_filtered:
            raise ValueError(f'No files found within ±1.5 hours of {date}')
    else:
        date_filtered = files
      
    sorted_files = sorted(date_filtered, key=lambda x: extract_date(x, keyword, file_type))
    
    datasets = []
    for filename in sorted_files:
        try:
            filepath = os.path.join(folder_path, filename)
            ds = xr.open_dataset(filepath)
            datasets.append(ds)
        except Exception as e:
            print(f'Error reading file {filename}: {str(e)}')
            continue
    
    if not datasets:
        raise ValueError(f'No valid {keyword} files found in the specified directory')
    
    combined_ds = xr.concat(datasets, dim='time')
    #close all the open datasets
    for ds in datasets:
        ds.close()
    combined_ds = convert_ds_time(combined_ds)
    
    return combined_ds.sortby('time')

def get_nearby_points_within_distance(latitudes, longitudes, reference_coords, 
                                    max_distance_km):
    """
    Find points within a specified distance of a reference point.
    
    Parameters
    ----------
    latitudes : array-like                   | Array of latitude values
    longitudes : array-like                  | Array of longitude values
    reference_coords : list                  | [latitude, longitude] of reference point
    max_distance_km : float                  | Maximum distance in kilometers
        
    Returns
    -------
    tuple
        (indices, shortest_distance, longest_distance, shortest_distance_index)
    """
    distance_array = np.zeros(len(latitudes))
    for i in range(len(latitudes)):
        coords_1 = [latitudes[i], longitudes[i]]
        coords_2 = reference_coords
        distance_array[i] = geopy.distance.geodesic(coords_1, coords_2).km
        
    distance_idx_nearest = np.where(distance_array < max_distance_km)
    if len(distance_idx_nearest[0]) < 2:
        print('Not enough points within the specified distance.')
        shortest_distance = None
        longest_distance = None
        shortest_distance_idx = None
    else:
        nearest_distances = distance_array[distance_idx_nearest]
        
        # All the following parameters refer to the cropped indices since the mask in 
        # line 389 is applied
        shortest_distance = np.min(nearest_distances)
        shortest_distance_idx = np.where(nearest_distances == shortest_distance)
        longest_distance = np.max(nearest_distances)

    return (distance_idx_nearest, shortest_distance, longest_distance, 
            shortest_distance_idx[0][0])

def load_process_scc_L1(sccpath):
    """
    Loads and merges all scc files in the folder, sorted by time.
    
    Parameters
    ----------
    sccpath : str        | Path to the ground station data folder
        
    Returns
    -------
    tuple
        (processed_data, station_name, station_coordinates)
    """

    scc = process_multiple_files(sccpath, 'EARLINET', 'HIRELPP')
    station_name = scc.attrs['location'].split(',')[0].strip()
    
    try:
        station_coordinates = [
            scc['latitude'].values[0],
            scc['longitude'].values[0]
        ]
    except ValueError:
        raise ValueError('Could not extract single values for coordinates')
        
    cropped_scc = scc.isel(channel=0, depolarization=0)
    return cropped_scc, station_name, station_coordinates
        


def load_crop_EC_product(filepath, station_coordinates, product, max_distance=50,
                        second_trim=False, second_distance=None):
    """
    Loads and trims EarthCARE products to desired distance around ground station.
    
    Parameters
    ----------
    filepath : str                      |Path to the EarthCARE product file
    station_coordinates : list          |[latitude, longitude] of the station
    product : str                       |Type of product to load ('ANOM', 'AEBD', or 'ATC')
    max_distance : float, optional      | Maximum distance in km for first trim
    second_trim : bool, optional        | Enable second trimming of dataset
    second_distance : float, optional   | Distance for second trim
        
    Returns
    -------
    tuple
        Various components depending on product type and trim options
    """
    valid_products = ['ANOM', 'AEBD', 'ATC', 'MRGR']
    if product not in valid_products:
        raise ValueError(f'Product must be one of {valid_products}')
        
    if not isinstance(station_coordinates, (list, tuple)) or len(station_coordinates) != 2:
        raise ValueError('station_coordinates must be a list/tuple of [latitude, longitude]')
        
    if second_trim and second_distance is None:
        raise ValueError('second_distance must be provided when second_trim is True')
        
    if product == 'ANOM':
        data = ecio.load_ANOM(filepath)
        data['sample_altitude'].values = data['sample_altitude'].values - data['geoid_offset'].values[:, np.newaxis]
    elif product == 'AEBD':
        data = ecio.load_AEBD(filepath)
        data['height'].values = data['height'].values - data['geoid_offset'].values[:, np.newaxis]
    elif product == 'MRGR':
        data = ecio.load_MRGR(filepath)
    else:
        data = ecio.load_ATC(filepath)
        data['height'].values = data['height'].values - data['geoid_offset'].values[:, np.newaxis]

    product_name=(ecio.load_EC_product(filepath, group='HeaderData/VariableProductHeader/MainProductHeader', 
                               trim=False))['productName'].item()
    baseline = (product_name.split('_')[1])[2:]
    
    if product == 'MRGR':
        threshold = 1e36
        idx = data['latitude'].sizes['across_track'] // 2
        latitude = (data['latitude'])#.where(data['latitude'] < threshold))#.mean(dim ='across_track')
        longitude = (data['longitude'])#.where(data['longitude'] < threshold)#.mean(dim ='across_track')
        distance_idx_nearest, s_dist, l_dist, s_dist_idx = get_nearby_points_within_distance(
            latitude.isel(across_track=idx),
            longitude.isel(across_track=idx),
            station_coordinates,
            max_distance_km=max_distance
        )
    else:
        distance_idx_nearest, s_dist, l_dist, s_dist_idx = get_nearby_points_within_distance(
            data['latitude'],
            data['longitude'],
            station_coordinates,
            max_distance_km=max_distance
        )
        
    cropped_data = data.isel(along_track=distance_idx_nearest[0])
    
    if product == 'ATC':
        return data, cropped_data, baseline
    if product == 'MRGR':
        return data, cropped_data, baseline
        
    time = cropped_data['time']
    shortest_time = time[s_dist_idx].values
        # Inside load_crop_EC_product
    if second_trim:
        # _2 to each name since they refer to the second trim product.
        distance_idx_nearest_2, s_dist_2, l_dist_2, s_dist_idx_2 = get_nearby_points_within_distance(
            data['latitude'],
            data['longitude'],
            station_coordinates,
            max_distance_km=second_distance
        )
        second_cropped_data = data.isel(along_track=distance_idx_nearest_2[0])
        return (data, cropped_data, shortest_time, baseline,
                distance_idx_nearest, s_dist, s_dist_idx, second_cropped_data)
                
    return (data, cropped_data, shortest_time, baseline,
            distance_idx_nearest, s_dist, s_dist_idx)

def read_pollynet_profile(file, data=False):
    """
    Read PollyNET netCDF  profile file and return datasets with EarthCARE-aligned names.
    
    Parameters
    ----------
    file : str                        | Path to file or xarray Dataset
    data : bool, optional             | Whether input is already a Dataset
        
    Returns
    -------
    tuple (ds_raman, ds_klett)       | Two datasets with aligned variable names
    """
    ds_orig = file if data else xr.open_dataset(file)
    
    raman_mapping = {
        'aerBsc_raman_355': 'particle_backscatter_coefficient_355nm',
        'uncertainty_aerBsc_raman_355': 'particle_backscatter_coefficient_355nm_error',
        'aerExt_raman_355': 'particle_extinction_coefficient_355nm',
        'uncertainty_aerExt_raman_355': 'particle_extinction_coefficient_355nm_error',
        'aerLR_raman_355': 'lidar_ratio_355nm',
        'uncertainty_aerLR_raman_355': 'lidar_ratio_355nm_error',
        'parDepol_raman_355': 'particle_linear_depol_ratio_355nm',
        'uncertainty_parDepol_raman_355': 'particle_linear_depol_ratio_355nm_error',
        'start_time':'start_time',
        'end_time': 'end_time'
    }
    
    klett_mapping = {
        'aerBsc_klett_355': 'particle_backscatter_coefficient_355nm',
        'uncertainty_aerBsc_klett_355': 'particle_backscatter_coefficient_355nm_error',
        'parDepol_klett_355': 'particle_linear_depol_ratio_355nm',
        'uncertainty_parDepol_klett_355': 'particle_linear_depol_ratio_355nm_error',
        'start_time':'start_time',
        'end_time': 'end_time'
    }
    
    raman_data = {}
    for old_name, new_name in raman_mapping.items():
        if old_name in ds_orig:
            raman_data[new_name] = (
                ds_orig[old_name].dims,
                ds_orig[old_name].values,
                ds_orig[old_name].attrs)
    
    ds_raman = xr.Dataset(
        raman_data,
        coords={
            'method': ds_orig['method'],
            'height': ds_orig['height'],
            'reference_height': ds_orig['reference_height']}
                        )
    
    klett_data = {}
    for old_name, new_name in klett_mapping.items():
        if old_name in ds_orig:
            klett_data[new_name] = (
                ds_orig[old_name].dims,
                ds_orig[old_name].values,
                ds_orig[old_name].attrs)
    
    ds_klett = xr.Dataset(
        klett_data,
        coords={
            'method': ds_orig['method'],
            'height': ds_orig['height'],
            'reference_height': ds_orig['reference_height']}
                    )
    
    for coord in ['method', 'height', 'reference_height']:
        if coord in ds_orig.coords and hasattr(ds_orig[coord], 'attrs'):
            ds_raman[coord].attrs = ds_orig[coord].attrs
            ds_klett[coord].attrs = ds_orig[coord].attrs
    
    return ds_raman, ds_klett

def read_scc_profile(file, time_idx):
    """
    Read PollyNET netCDF profile file and return datasets with EarthCARE-aligned names.
    
    Parameters
    ----------
    file : str                        | Path to file or xarray Dataset
    data : bool, optional             | Whether input is already a Dataset
        
    Returns
    -------
    tuple (ds_raman, ds_klett)       | Two datasets with aligned variable names
    """
    old_klett = None
    old_raman = None
    
    if file[0] is not None:
        old_klett = file[0].isel(time=time_idx, wavelength=0)
    if file[1] is not None:
        old_raman = file[1].isel(time=time_idx, wavelength=0)
    
    raman_mapping = {
        'backscatter': 'particle_backscatter_coefficient_355nm',
        'error_backscatter': 'particle_backscatter_coefficient_355nm_error',
        'extinction': 'particle_extinction_coefficient_355nm',
        'error_extinction': 'particle_extinction_coefficient_355nm_error',
        'lidarratio': 'lidar_ratio_355nm',
        'error_lidarratio': 'lidar_ratio_355nm_error',
        'altitude': 'height'
    }
    
    klett_mapping = {
        'backscatter': 'particle_backscatter_coefficient_355nm',
        'error_backscatter': 'particle_backscatter_coefficient_355nm_error',
        'particledepolarization': 'particle_linear_depol_ratio_355nm',
        'error_particledepolarization': 'particle_linear_depol_ratio_355nm_error',
        'altitude': 'height'
    }
    
    # Initialize variables
    ds_klett = None
    ds_raman = None
    pardepol = None
    pardepol_er = None
    
    # Process Klett data first
    if old_klett is not None:
        if 'raman_backscatter_algorithm' in old_klett.data_vars:
            # Special case: extract depolarization data for later use with Raman
            pardepol = old_klett['particledepolarization']
            pardepol_er = old_klett['error_particledepolarization']
            # Don't create ds_klett in this case
        else:
            # Normal Klett processing
            klett_data = {}
            for old_name, new_name in klett_mapping.items():
                if old_name in old_klett:
                    klett_data[new_name] = (
                        old_klett[old_name].dims,
                        old_klett[old_name].values,
                        old_klett[old_name].attrs)
            
            ds_klett = xr.Dataset(
                klett_data,
                coords={
                    'time': old_klett['time'],
                    'nv': old_klett['nv'],
                    'altitude': old_klett['altitude'],
                    'wavelength': old_klett['wavelength']
                })
            for coord in ['method', 'height', 'reference_height']:
                if coord in old_klett.coords and hasattr(old_klett[coord], 'attrs'):
                    ds_klett[coord].attrs = old_klett[coord].attrs
    
    # Process Raman data 
    if old_raman is not None:            
        raman_data = {}
        for old_name, new_name in raman_mapping.items():
            if old_name in old_raman:
                raman_data[new_name] = (
                    old_raman[old_name].dims,
                    old_raman[old_name].values,
                    old_raman[old_name].attrs)
        
        ds_raman = xr.Dataset(
            raman_data,
            coords={
                'time': old_raman['time'],
                'nv': old_raman['nv'],
                'altitude': old_raman['altitude'],
                'wavelength': old_raman['wavelength']
            })
        
        for coord in ['method', 'height', 'reference_height']:
            if coord in old_raman.coords and hasattr(old_raman[coord], 'attrs'):
                ds_raman[coord].attrs = old_raman[coord].attrs
                
        # Add the depolarization data to ds_raman if we have it and no ds_klett
        if ds_klett is None and pardepol is not None and pardepol_er is not None:
            ds_raman['particle_linear_depol_ratio_355nm'] = pardepol
            ds_raman['particle_linear_depol_ratio_355nm_error'] = pardepol_er
    
    # Return the datasets
    return ds_raman, ds_klett
def cut_gnd_noise(sat_ds, gnd_ds, variables, heightvar_EC='height', 
                  heightvar_gnd='height', step=50, threshold=200):
    """
    Function to cut noisy ground data. For a range of every 50km checks the average 
    between the datasets and if their difference is above a threshold, it fills with nan 
    values the respective points.
    
    Parameters
    ----------
    sat_ds : xarray.Dataset
        Satellite dataset containing the reference measurements
    gnd_ds : xarray.Dataset
        Ground dataset to be filtered
    variables : str or list of str
        Variable name(s) to process
    heightvar_EC : str, optional
        Name of height coordinate in satellite (EC) dataset. Default is 'height'
    heightvar_gnd : str, optional
        Name of height coordinate in ground dataset. Default is 'height'
    step : int, optional
        Step size in kilometers for comparison windows. Default is 50.
    threshold : float, optional
        Maximum allowed percentage difference between averages. Default is 200.
        
    Returns
    -------
    xarray.Dataset
        Filtered ground dataset with noise replaced by NaN values
    """
    import numpy as np
    import xarray as xr
    
    # Convert single variable to list
    if isinstance(variables, str):
        variables = [variables]
    
    # Create a copy of ground dataset to avoid modifying the original
    filtered_gnd = gnd_ds.copy(deep=True)
    
    # Verify variables exist in both datasets
    for var in variables:
        if var not in sat_ds or var not in gnd_ds:
            raise ValueError(f"Variable {var} not found in both datasets")
    
    # Get the height coordinates
    heights_gnd = gnd_ds[heightvar_gnd].values
    heights_EC = sat_ds[heightvar_EC].values
    
    # Calculate the window ranges using the overlapping height range
    min_height = max(heights_gnd.min(), heights_EC.min())
    max_height = min(heights_gnd.max(), heights_EC.max())
    height_ranges = np.arange(min_height, max_height + step, step)
    
    # Process each specified variable
    for var in variables:
        # Iterate through each height range
        for start_height in height_ranges[:-1]:
            end_height = start_height + step
            
            try:
                # Select data within the current height range
                sat_mask = (sat_ds[heightvar_EC] >= start_height) & (sat_ds[heightvar_EC] < end_height)
                gnd_mask = (gnd_ds[heightvar_gnd] >= start_height) & (gnd_ds[heightvar_gnd] < end_height)
                
                sat_slice = sat_ds[var].where(sat_mask)
                gnd_slice = gnd_ds[var].where(gnd_mask)
                
                # Check if we have any valid data in this range
                if sat_slice.count() == 0 or gnd_slice.count() == 0:
                    continue
                
                # Calculate averages for the window
                sat_avg = float(sat_slice.mean(skipna=True))
                gnd_avg = float(gnd_slice.mean(skipna=True))
                # Calculate percentage difference
                if sat_avg != 0 and not np.isnan(sat_avg) and not np.isnan(gnd_avg):
                    pct_diff = abs((gnd_avg - sat_avg) / sat_avg * 100)
                    
                    # If difference exceeds threshold, replace values with NaN
                    if pct_diff > threshold:
                        filtered_gnd[var] = filtered_gnd[var].where(
                            ~((filtered_gnd[heightvar_gnd] >= start_height) & 
                              (filtered_gnd[heightvar_gnd] < end_height)), 
                            np.nan)
            
            except Exception as e:
                print(f"Error processing {var} at height range {start_height}-{end_height}: {e}")
                continue
                    
    return filtered_gnd

def read_sula_file(data_path):
    """
    Reads sula file

    Parameters
    ----------
    data_path : str         | Path to Thelysis files

    Returns
    -------
    tuple | (ds, statio_coords, station_name)
               Ground station dataset, station coordinates, station name

    """
    ds = process_multiple_files(data_path, network='THELISYS')
    
    station_name = 'Thelysis'
    station_coords = ([(ds['LATITUDE'].item()),(ds['LONGITUDE'].item())])

    return ds, station_coords, station_name

def process_sula_profile(file, data=False):
    """
    Read sula netCDF  profile file and return datasets with EarthCARE-aligned names.
    
    Parameters
    ----------
    file : str                        | Path to file or xarray Dataset
    data : bool, optional             | Whether input is already a Dataset
        
    Returns
    -------
    tuple (ds_raman, ds_klett)       | Two datasets with aligned variable names
    """
    ds_orig = file if data else xr.open_dataset(file)
    
    #ds_orig = ds_orig.isel(time=0)
    raman_mapping = {
        'RB355': 'particle_backscatter_coefficient_355nm',
        'RB355_ERROR': 'particle_backscatter_coefficient_355nm_error',
        'EXT355': 'particle_extinction_coefficient_355nm',
        'EXT355_ERROR': 'particle_extinction_coefficient_355nm_error',
        'LR355': 'lidar_ratio_355nm',
        'LR355_ERROR': 'lidar_ratio_355nm_error',
        'START_TIME':'start_time',
        'END_TIME': 'end_time'
    }
    
    klett_mapping = {
        'KB355': 'particle_backscatter_coefficient_355nm',
        'KB355_ERROR': 'particle_backscatter_coefficient_355nm_error',

        'START_TIME':'start_time',
        'END_TIME': 'end_time'
    }
    
    raman_data = {}
    for old_name, new_name in raman_mapping.items():
        if old_name in ds_orig:
            raman_data[new_name] = (
                ds_orig[old_name].dims,
                ds_orig[old_name].values,
                ds_orig[old_name].attrs)
    
    ds_raman = xr.Dataset(
        raman_data,
        coords={
            'height': ds_orig['altitude'],
            'START_TIME': ds_orig['START_TIME'],
            'END_TIME': ds_orig['END_TIME']}
                        )
    
    klett_data = {}
    for old_name, new_name in klett_mapping.items():
        if old_name in ds_orig:
            klett_data[new_name] = (
                ds_orig[old_name].dims,
                ds_orig[old_name].values,
                ds_orig[old_name].attrs)
    
    ds_klett = xr.Dataset(
        klett_data,
        coords={
            'height': ds_orig['altitude'],
            'START_TIME': ds_orig['START_TIME'],
            'END_TIME': ds_orig['END_TIME']}
                        )
    
    # Handle depolarization separately due to wavelength conversion
    if 'PLDR532' in ds_orig.variables:
        # Constants for conversion
        convfactor_dp355 = 1
        convfactor_dp355_err = 0
        
        # Get original data
        pdr532 = ds_orig.variables['PLDR532'].values
        pdr532_err = ds_orig.variables['PLDR532_ERROR'].values        
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
        ds_raman['particle_linear_depol_ratio_355nm'] = ('altitude', pdr355_raman)
        ds_raman['particle_linear_depol_ratio_355nm_error'] = ('altitude', pdr355_err)
        ds_klett['particle_linear_depol_ratio_355nm'] = ('altitude', pdr355_klett)
        ds_klett['particle_linear_depol_ratio_355nm_error'] = ('altitude', pdr355_err)
        
        # Add metadata
        ds_raman['particle_linear_depol_ratio_355nm'].attrs['units'] = 'ratio'
        ds_raman['particle_linear_depol_ratio_355nm'].attrs['wavelength'] = '355nm (converted from 532nm)'
        ds_klett['particle_linear_depol_ratio_355nm'].attrs['units'] = 'ratio'
        ds_klett['particle_linear_depol_ratio_355nm'].attrs['wavelength'] = '355nm (converted from 532nm)'
    
    return ds_raman, ds_klett

