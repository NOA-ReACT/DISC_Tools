#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing module for EarthCARE analysis tools.
"""

import os
import glob
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import geopy.distance
from ectools_noa import ecio

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

def build_paths(root_dir, level):
    """
    Build paths dictionary from root directory
    
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

    if level =='L1':
        paths = {
            'ANOM': glob.glob(os.path.join(root_dir, level, 'eca', '*ATL_NOM*.h5')),
            'SIM': glob.glob(os.path.join(root_dir, level, 'sim', '*ATL_NOM*.h5')),
            'SCC': os.path.join(root_dir, level, 'gnd', 'scc'),
            'POLLY': os.path.join(root_dir, level, 'gnd', 'tropos'),
            'OUTPUT': os.path.join(root_dir, level, 'plots_comparison')
            }
        paths['ANOM'] =paths['ANOM'][0] if paths['ANOM'] else None
        paths['SIM'] =paths['SIM'][0] if paths['SIM'] else None

    elif level == 'L2':      
        paths = {
            'AEBD': glob.glob(os.path.join(root_dir, level, 'eca', '*ATL_EBD*.h5')),
            'ATC': glob.glob(os.path.join(root_dir, level, 'eca', '*ATL_TC__*.h5')),
            'SCC': os.path.join(root_dir, level, 'gnd', 'scc'),
            'POLLY': os.path.join(root_dir, level, 'gnd', 'tropos'),
            'OUTPUT': os.path.join(root_dir, level, 'plots_comparison')
            }
    
        # Take first file if list (for AEBD and ATC)
        paths['AEBD'] = paths['AEBD'][0] if paths['AEBD'] else None
        paths['ATC'] = paths['ATC'][0] if paths['ATC'] else None
    
    # Create OUTPUT directory if it doesn't exist
    os.makedirs(paths['OUTPUT'], exist_ok=True)
    
    return paths

def load_ground_data(network, pollypath, sccpath, data_type='L1'):
    """
    Load ground-based lidar data based on network type and data requirements.
    
    Parameters:
    -----------
    network : str        | Network type ('EARLINET' or 'POLLYXT')
    path : str           | Path to data files
    data_type : str      | Type of data processing required ('L1' or 'L2')
    
    Returns:
    --------
    tuple        | (gnd_quicklook, gnd_profile (optional), station_name, station_coordinates)
    """

    if network == 'EARLINET':
        if data_type == 'L1':
            # L1 processing
            gnd_quicklook, station_name, station_coordinates = load_process_scc_L1(sccpath)
        else:
            # L2 processing
            gnd_profile_a = process_multiple_files(sccpath, 'EARLINET', 'b0355')
            gnd_profile_b = process_multiple_files(sccpath, 'EARLINET', 'e0355')
            gnd_profile = xr.merge([gnd_profile_a, gnd_profile_b])
            gnd_quicklook, station_name, station_coordinates = load_process_scc_L1(sccpath)

    elif network == 'POLLYXT':
        if data_type == 'L1':
            # L1 processing
            gnd_quicklook_a = process_multiple_files(pollypath, 'POLLYXT', 'att_bsc')
            gnd_quicklook_b = process_multiple_files(pollypath, 'POLLYXT', 'vol_depol')
            gnd_quicklook = xr.merge([gnd_quicklook_a, gnd_quicklook_b])
        else:
            # L2 processing
            gnd_quicklook = process_multiple_files(pollypath, 'POLLYXT', 'quasi_results_V2')
            # The following line needed for the quasi files to be plotted correctly
            # else they are squeezed between 0-3km 
            gnd_quicklook = gnd_quicklook.assign_coords(height=gnd_quicklook.height * 10)
            gnd_profile = process_multiple_files(pollypath, 'POLLYXT', 'profile')
            
        station_name, station_coordinates = get_polly_station_info(gnd_quicklook, data=True)
              
    else:
        raise ValueError(f"Unsupported network: {network}. Must be either 'EARLINET' or 'POLLYXT'")

    if data_type == 'L2':
        return gnd_quicklook, gnd_profile, station_name, station_coordinates
    return gnd_quicklook, station_name, station_coordinates
    
def get_polly_station_info(filename, data= False):
    """
    Read NetCDF file and return station name and coordinates.
    
    Parameters:
    -----------
    filename : str | Path to the NetCDF file
    data: bool     | Weather filename is the data file, or the filepath    
        
    Returns:
    --------
    station : str      | station_coordinates
    """
    try:
        
        if data:
            ds=filename
        else:
            ds = xr.open_dataset(filename)
        
        # Extract station name from global attributes
        station = ds.attrs.get('location')
        
        # Extract coordinates
        # Note: Using .item() to convert from numpy array to Python scalar
        lat = round(ds['latitude'].item(0),2)
        lon = round(ds['longitude'].item(0),2)
        
        station_coordinates = [lat,lon]

        # Close the dataset
        ds.close()
        
        return station, station_coordinates
        
    except Exception as e:
        raise Exception(f"Error reading NetCDF file: {str(e)}")
        
def convert_ds_time(ds):
    """
    Convert the time coordinate of quasi dataset from Unix timestamps 
    to datetime64[ns] format to match SCC format.
    
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
        
def process_multiple_files(folder_path, network, file_type=None):
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
        
    Returns
    -------
    xarray.Dataset
        Combined dataset
    """
    if network == 'EARLINET':
        keyword = 'ECVT'
    elif network == 'POLLYXT':
        keyword = 'NOA'
    else: 
        raise ValueError('Only EARLINET and POLLYXT networks available for processing')
       
    all_files = os.listdir(folder_path)
    
    if network == 'EARLINET':
        files = [f for f in all_files if file_type in f]
    elif network == 'POLLYXT':
        files = [f for f in all_files if file_type in f]

    if not files:
        raise ValueError(f'No {network} files found in {folder_path}')
    
    sorted_files = sorted(files, key=lambda x: extract_date(x, keyword, file_type))
    
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
        return distance_idx_nearest, None
        
    nearest_distances = distance_array[distance_idx_nearest]
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
    try:
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
        
    except Exception as e:
        raise RuntimeError(f'Error processing SCC files: {str(e)}')

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
    valid_products = ['ANOM', 'AEBD', 'ATC']
    if product not in valid_products:
        raise ValueError(f'Product must be one of {valid_products}')
        
    if not isinstance(station_coordinates, (list, tuple)) or len(station_coordinates) != 2:
        raise ValueError('station_coordinates must be a list/tuple of [latitude, longitude]')
        
    if second_trim and second_distance is None:
        raise ValueError('second_distance must be provided when second_trim is True')
        
    if product == 'ANOM':
        data = ecio.load_ANOM(filepath)
    elif product == 'AEBD':
        data = ecio.load_AEBD(filepath)
    else:
        data = ecio.load_ATC(filepath)
    
    product_name=(ecio.load_EC_product(filepath, group='HeaderData/VariableProductHeader/MainProductHeader', 
                               trim=False))['productName'].item()
    baseline = (product_name.split('_')[1])[2:]
    
    distance_idx_nearest, s_dist, l_dist, s_dist_idx = get_nearby_points_within_distance(
        data['latitude'],
        data['longitude'],
        station_coordinates,
        max_distance_km=max_distance
    )
    
    cropped_data = data.isel(along_track=distance_idx_nearest[0])
    
    if product == 'ATC':
        return data, cropped_data, baseline
        
    time = cropped_data['time']
    shortest_time = time[s_dist_idx].values
    
    if second_trim:
        distance_idx_nearest, s_dist, l_dist, s_dist_idx = get_nearby_points_within_distance(
            data['latitude'],
            data['longitude'],
            station_coordinates,
            max_distance_km=second_distance
        )
        second_cropped_data = data.isel(along_track=distance_idx_nearest[0])
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
    tuplε (ds_raman, ds_klett)       | Two datasets with aligned variable names
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
        'uncertainty_parDepol_raman_355': 'particle_linear_depol_ratio_355nm_error'
    }
    
    klett_mapping = {
        'aerBsc_klett_355': 'particle_backscatter_coefficient_355nm',
        'uncertainty_aerBsc_klett_355': 'particle_backscatter_coefficient_355nm_error',
        'parDepol_klett_355': 'particle_linear_depol_ratio_355nm',
        'uncertainty_parDepol_klett_355': 'particle_linear_depol_ratio_355nm_error'
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

