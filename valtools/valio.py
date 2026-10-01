#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing module for EarthCARE analysis tools.

"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> ectools_noa
import glob
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import geopy.distance

from ectools_noa import ecio
from local_reader import read_RV_meteor#, process_sula_profile

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

    
def filter_files_by_baseline(file_list, baseline):
    """Filter files based on baseline criteria"""
    if not baseline or not file_list:
        return file_list[0] if file_list else None
    
    if baseline.endswith('*'):
        # Wildcard matching (e.g., 'A*' matches 'AA', 'AB', 'AC', etc.)
        prefix = baseline[:-1]
        filtered_files = [f for f in file_list if f'EX{prefix}' in os.path.basename(f)]
    else:
        # Exact baseline matching (e.g., 'AE' matches 'EXAE')
        filtered_files = [f for f in file_list if f'EX{baseline}' in os.path.basename(f)]
    
    return filtered_files[0] if filtered_files else None
    
def build_paths(root_dir, network, level, baseline=None):
    """
    Build paths dictionary from root directory and create directories if they don't exist

    Optional convenience for one example folder structure (see README,
    'Example folder structure'). Not required: the normal way is to give the
    file paths directly in CUSTOM_PATHS_L1 / CUSTOM_PATHS_L2 (valconfig.py).
    
    Parameters
    ----------
    root_dir : str      | Root directory containing the L1 and L2 structure
    network : str       | Network type (EARLINET, THELISYS, POLLYXT, LICHT)
    level: str          | Level of data process to search for the equivalent folder: 
                         either L1 or L2
    baseline: str       | Baseline filter (e.g., 'AE', 'BA', 'A*', 'B*', None)
                         If None, returns first file found (original behavior)
                         If specific (e.g., 'AE'), filters for files containing 'EXAE'
                         If wildcard (e.g., 'A*'), filters for all baselines starting with 'A'
        
    Returns
    -------
    dict
        Dictionary with paths for AEBD, ATC, SCC, POLLY, and OUTPUT
    """
    # Ensure root_dir exists
    if not os.path.exists(root_dir):
        raise ValueError(f"Root directory does not exist: {root_dir}")
    
    if network == 'EARLINET' or network == 'THELISYS':
        gnd_suffix = 'scc'
    elif network == 'POLLYXT':
        gnd_suffix = 'tropos'
    elif network == 'LICHT':
        gnd_suffix = 'licht'

    # First create the base directories
    if level == 'L1':
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
    if level == 'L1':
        anom_files = glob.glob(os.path.join(base_dirs['ANOM'], '*ATL_NOM*.h5'))
        sim_files = glob.glob(os.path.join(base_dirs['SIM'], '*ATL_NOM*.h5'))
        
        paths = {
            'ANOM': filter_files_by_baseline(anom_files, baseline),
            'SIM': sim_files[0] if sim_files else None,
            'GND': base_dirs['GND'],
            'OUTPUT': base_dirs['OUTPUT']
        }
        
    elif level == 'L2':
        aebd_files = glob.glob(os.path.join(base_dirs['AEBD'], '*ATL_EBD*.h5'))
        atc_files = glob.glob(os.path.join(base_dirs['ATC'], '*ATL_TC__*.h5'))
        actc_files = glob.glob(os.path.join(base_dirs['ACTC'], '*AC__TC__*'))
        mrgr_files = glob.glob(os.path.join(base_dirs['MRGR'], '*MSI_RGR_*'))
        cfmr_files = glob.glob(os.path.join(base_dirs['CFMR'], '*CPR_FMR*'))
        
        paths = {
            'AEBD': filter_files_by_baseline(aebd_files, baseline),
            'ATC': filter_files_by_baseline(atc_files, baseline),
            'ACTC': filter_files_by_baseline(actc_files, baseline),
            'MRGR': filter_files_by_baseline(mrgr_files, baseline),
            'CFMR': filter_files_by_baseline(cfmr_files, baseline),
            'GND': base_dirs['GND'],
            'OUTPUT': base_dirs['OUTPUT']
        }
    
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
                gnd_quicklook, _, _ = load_process_scc_L1(data_path)
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
        try:
            gnd_quicklook, station_name1, station_coordinates1 = load_process_scc_L1(data_path)
        except Exception:
            gnd_quicklook = None   
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
                if (filedate > date - pd.Timedelta('4H')) & (filedate < date + pd.Timedelta('4H')):
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

def filter_dataset_by_values(ds, filter_var, filter_values, variables_to_filter=None,
                              fill_value=None, product='EC', bad_percentage=1.0, hmax=None):
    """
    Mask dataset values where `filter_var` takes any of `filter_values`.

    Two levels (EC products):
      1. pixel   : bins with a bad value are set to fill_value
      2. profile : whole profiles are set to fill_value if the fraction of bad
                   bins exceeds bad_percentage (computed below hmax, if given)

    Parameters
    ----------
    ds : xarray.Dataset                         | Input dataset
    filter_var : str                            | Variable holding the filter criteria (e.g. 'quality_status')
    filter_values : list/tuple/set or scalar    | Values to mask (e.g. [2, 3, 4])
    variables_to_filter : list of str, optional | Variables to mask. None -> all data variables
    fill_value : int, float or None             | Value for masked data. None -> NaN (use None before averaging)
    product : str                               | 'EC' enables the profile-level mask (needs 'JSG_height')
    bad_percentage : float                      | Profile removed if bad fraction > this (1.0 -> never)
    hmax : float, optional                      | Only bins below hmax (m) count for the bad fraction

    Returns
    -------
    xarray.Dataset
        Dataset with the selected variables masked
    """
    if not isinstance(filter_values, (list, tuple, set)):
        filter_values = [filter_values]

    mask = ds[filter_var].isin(filter_values)
    print(f"QS filter: {int(mask.sum())}/{mask.size} bins masked "
          f"({float(mask.mean())*100:.1f}%)")

    if product == 'EC' and bad_percentage < 1.0:
        valid = ds[filter_var].notnull()
        if hmax is not None:
            below = ds['height'] <= hmax
            valid, bad = valid & below, mask & below
        else:
            bad = mask
        bad_fraction = bad.sum(dim='JSG_height') / valid.sum(dim='JSG_height').clip(min=1)
        profile_mask = bad_fraction > bad_percentage          # (along_track,)
        print(f"QS filter: {int(profile_mask.sum())}/{profile_mask.size} profiles removed "
              f"with >{bad_percentage*100:.0f}% bad bins")
        mask = mask | profile_mask                             # broadcast by dim name

    exclude_vars = {'height', 'time', 'lat', 'lon', 'latitude', 'longitude',
                    'simple_classification', filter_var}
    if variables_to_filter is not None:
        if isinstance(variables_to_filter, str):
            variables_to_filter = [variables_to_filter]
        vars_to_process = [v for v in variables_to_filter if v in ds.data_vars]
    else:
        vars_to_process = [v for v in ds.data_vars
                           if v not in exclude_vars and v not in ds.coords]

    out = ds.copy()
    for var in vars_to_process:
        # only mask variables that share the filter dimensions
        if set(mask.dims) <= set(ds[var].dims):
            out[var] = ds[var].where(~mask) if fill_value is None else ds[var].where(~mask, fill_value)
    return out


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
        print('Not enough points within the specified distance. new')
        return (distance_idx_nearest, None, None, None)
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
                         second_trim=False, second_distance=None, data=True,
                         qs_var='quality_status', qs_filter=None, qs_bad_fraction=1.0, qs_hmax=None):
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
    data : bool or xr.Dataset, optional | True (default) -> load the product from
                                          `filepath` and apply the geoid correction.
                                          Pass an already-opened, already-geoid-
                                          corrected Dataset to skip reading.
    qs_var : str, optional              | Quality variable used for filtering: 'quality_status'
                                          (default) or 'extended_data_quality_status'
    qs_filter : list, optional          | qs_var values to mask in the cropped data
                                          (e.g. [2, 3, 4]). None -> no filtering. Skipped for
                                          products without qs_var (e.g. ANOM)
    qs_bad_fraction : float, optional   | Remove whole profiles with more bad bins than this
                                          fraction (1.0 -> never)
    qs_hmax : float, optional           | Bad fraction counted only below this height (m);
                                          the masking itself covers the whole profile

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

    load_from_file = (data is True)

    if not load_from_file:
        if not isinstance(data, xr.Dataset):
            raise TypeError('data must be True or an already-opened xarray.Dataset')
        if filepath is None:
            raise ValueError('filepath is still required to read the product baseline')
    else:
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

    product_name = (ecio.load_EC_product(filepath, group='HeaderData/VariableProductHeader/MainProductHeader',
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

    apply_qs = qs_filter is not None and qs_var in data
    if qs_filter is not None and not apply_qs:
        print(f"QS filter skipped: {product} has no '{qs_var}'")

    if apply_qs:
        cropped_data = filter_dataset_by_values(cropped_data, qs_var, qs_filter,
                                                bad_percentage=qs_bad_fraction, hmax=qs_hmax)

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
        if apply_qs:
            second_cropped_data = filter_dataset_by_values(second_cropped_data, qs_var, qs_filter,
                                                           bad_percentage=qs_bad_fraction, hmax=qs_hmax)
        return (data, cropped_data, shortest_time, baseline,
                distance_idx_nearest, s_dist, s_dist_idx, second_cropped_data)

    return (data, cropped_data, shortest_time, baseline,
            distance_idx_nearest, s_dist, s_dist_idx)

def read_pollynet_profile(file, data=False, wavelengths=('355', '532', '1064')):
    """
    Read PollyNET netCDF profile file and return datasets with
    EarthCARE-aligned names.

    Parameters
    ----------
    file : str or xarray.Dataset
        Path to file or already opened Dataset.

    data : bool, optional
        Whether input is already a Dataset.

    wavelengths : tuple of str
        Wavelengths to map. Variables absent from the file
        are skipped silently.

    Returns
    -------
    ds_raman, ds_klett : xarray.Dataset
        Raman and Klett datasets using the common internal naming convention.
        Singleton PollyNET-specific dimensions ('method', 'reference_height')
        are removed.
    """

    ds_orig = file if data else xr.open_dataset(file)

    def _build_mapping(method):
        """method is 'raman' or 'klett'."""

        mapping = {
            'start_time': 'start_time',
            'end_time': 'end_time'
        }

        for w in wavelengths:

            # Backscatter
            mapping[f'aerBsc_{method}_{w}'] = \
                f'particle_backscatter_coefficient_{w}nm'

            mapping[f'uncertainty_aerBsc_{method}_{w}'] = \
                f'particle_backscatter_coefficient_{w}nm_error'

            # Particle depolarization
            mapping[f'parDepol_{method}_{w}'] = \
                f'particle_linear_depol_ratio_{w}nm'

            mapping[f'uncertainty_parDepol_{method}_{w}'] = \
                f'particle_linear_depol_ratio_{w}nm_error'
            # Volume depolarization
            mapping[f'volDepol_{method}_{w}'] = \
                f'volume_linear_depol_ratio_{w}nm'
            
            mapping[f'uncertainty_volDepol_{method}_{w}'] = \
                f'volume_linear_depol_ratio_{w}nm_error'

            # Raman-only variables
            if method == 'raman':

                mapping[f'aerExt_{method}_{w}'] = \
                    f'particle_extinction_coefficient_{w}nm'

                mapping[f'uncertainty_aerExt_{method}_{w}'] = \
                    f'particle_extinction_coefficient_{w}nm_error'

                mapping[f'aerLR_{method}_{w}'] = \
                    f'lidar_ratio_{w}nm'

                mapping[f'uncertainty_aerLR_{method}_{w}'] = \
                    f'lidar_ratio_{w}nm_error'

        return mapping

    raman_mapping = _build_mapping('raman')
    klett_mapping = _build_mapping('klett')

    coords = {
        'height': ds_orig['height']
    }
    def _build_dataset(mapping):

        var_data = {}

        for old_name, new_name in mapping.items():

            if old_name in ds_orig:

                var_data[new_name] = (
                    ds_orig[old_name].dims,
                    ds_orig[old_name].values,
                    ds_orig[old_name].attrs
                )

        return xr.Dataset(var_data, coords=coords)

    ds_raman = _build_dataset(raman_mapping)
    ds_klett = _build_dataset(klett_mapping)

    if 'height' in ds_orig.coords:
        ds_raman['height'].attrs = ds_orig['height'].attrs
        ds_klett['height'].attrs = ds_orig['height'].attrs

    # ---------------------------------------------------------
    # Remove PollyNET storage dimensions that contain
    # only one retrieval solution.
    # ---------------------------------------------------------

    for dim in ['method', 'reference_height']:

        if ds_raman.sizes.get(dim) == 1:
            ds_raman = ds_raman.squeeze(dim=dim, drop=True)

        if ds_klett.sizes.get(dim) == 1:
            ds_klett = ds_klett.squeeze(dim=dim, drop=True)

    return ds_raman, ds_klett

def _add_time_window(ds, old):
    """Copy the measurement window of an SCC profile into start_time / end_time.

    Uses time_bounds (start, end) when present; otherwise start_time = time
    and no end_time.
    """
    if 'time_bounds' in old:
        bounds = old['time_bounds'].values.ravel()
        ds['start_time'] = ((), bounds[0])
        ds['end_time'] = ((), bounds[-1])
    else:
        ds['start_time'] = ((), old['time'].values)
    return ds


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
            ds_klett = _add_time_window(ds_klett, old_klett)
    
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
        ds_raman = _add_time_window(ds_raman, old_raman)
                
        # Add the depolarization data to ds_raman if we have it and no ds_klett
        if ds_klett is None and pardepol is not None and pardepol_er is not None:
            ds_raman['particle_linear_depol_ratio_355nm'] = pardepol
            ds_raman['particle_linear_depol_ratio_355nm_error'] = pardepol_er
    
    # Return the datasets
    return ds_raman, ds_klett
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