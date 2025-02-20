import xarray as xr
import numpy as np

import os
from glob import glob

def drop_dim(ds, dimname):
    return ds.drop([v for v in ds.data_vars if dimname in ds[v].dims])

frame_limits = {'A': [-22.5,  22.5], 'B': [ 22.5,  67.5], 'C':[ 67.5,  67.5], 'D':[ 67.5,  22.5], 
                'E': [ 22.5, -22.5], 'F': [-22.5, -67.5], 'G':[-67.5, -67.5], 'H':[-67.5, -22.5]}

get_frame = lambda ds: ds.encoding['source'].split('/')[-1].split('.')[0].split("_")[-1][-1]

def get_frame_alongtrack(ds, along_track_dim='along_track', latvar='latitude'):    
    lat_framestart, lat_framestop = frame_limits[get_frame(ds)]
    
    i_halfway = len(ds[along_track_dim])//2
    i_framestart = np.argmin(np.abs(ds[latvar].values[:i_halfway] - lat_framestart))
    i_framestop  = i_halfway + np.argmin(np.abs(ds[latvar].values[i_halfway:] - lat_framestop))
    
    return i_framestart, i_framestop

def trim_to_frame(ds, along_track_dim='along_track', latvar='latitude', sel_nadir=None):
    if sel_nadir:
        """
        sel_nadir = {'across_track':284}
        """
        return ds.isel({along_track_dim:slice(*get_frame_alongtrack(ds.isel(sel_nadir), along_track_dim, latvar))})
    else:
        return ds.isel({along_track_dim:slice(*get_frame_alongtrack(ds, along_track_dim, latvar))})

def load_AEBD(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_ATL_EBD*/ECA_EXAA_ATL_EBD_*.h5"))[-1]
    AEBD = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        AEBD = trim_to_frame(AEBD)
    return AEBD

def load_AAER(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_ATL_AER_*/ECA_EXAA_ATL_AER_*.h5"))[-1]
    AAER = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        AAER = trim_to_frame(AAER)
    return AAER

def load_AICE(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_ATL_ICE_*/ECA_EXAA_ATL_ICE_*.h5"))[-1]
    AICE = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        AICE = trim_to_frame(AICE)
    return AICE 

def load_ATC(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_ATL_TC__*/ECA_EXAA_ATL_TC__*.h5"))[-1]
    ATC = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        ATC = trim_to_frame(ATC)
    return ATC
    
def load_CFMR(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_CPR_FMR_*/ECA_EXAA_CPR_FMR_*.h5"))[-1]
    CFMR = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        CFMR = trim_to_frame(CFMR)
    return CFMR  

def load_CCD(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_CPR_CD__*/ECA_EXAA_CPR_CD__*.h5"))[-1]
    CCD = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        CCD = trim_to_frame(CCD)
    return CCD    
    
def load_CTC(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_CPR_TC__*/ECA_EXAA_CPR_TC__*.h5"))[-1]
    CTC = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        CTC = trim_to_frame(CTC)
    return CTC

def load_CCLD(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_CPR_CLD_*/ECA_EXAA_CPR_CLD_*.h5"))[-1]
    CCLD = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        CCLD = trim_to_frame(CCLD)
    return CCLD 


def load_ACTC(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_AC__TC__*/ECA_EXAA_AC__TC__*.h5"))[-1]
    ACTC = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        ACTC = trim_to_frame(ACTC)
    return ACTC





def load_ACMCOM(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_ACM_COM_*/ECA_EXAA_ACM_COM_*.h5"))[-1]
    ACMCOM = xr.open_dataset(srcfile, group='ScienceData').sel(atmosphere=0)
    if trim:
        ACMCOM = trim_to_frame(ACMCOM, latvar='latitude_active')
    return drop_dim(ACMCOM, 'across_track').rename({'latitude_active':'latitude', 'longitude_active':'longitude'})

def load_ACMCAP(srcdir, trim=True):
    srcfile = sorted(glob(f"{srcdir}/output/ECA_EXAA_ACM_CAP_*/ECA_EXAA_ACM_CAP_*.h5"))[-1]
    ACMCAP = xr.open_dataset(srcfile, group='ScienceData')
    if trim:
        ACMCAP = trim_to_frame(ACMCAP)
    return  ACMCAP


def load_ANOM(srcdir, trim=True):
    ANOM_srcfile = sorted(glob(f"{srcdir}/input/ECA_EXAA_ATL_NOM_1B_*/ECA_EXAA_ATL_NOM_1B_*.h5"))[-1]
    ANOM = xr.open_dataset(ANOM_srcfile, group='ScienceData')
    if trim:
        ANOM = trim_to_frame(ANOM, latvar='ellipsoid_latitude')
    return ANOM

def load_CNOM(srcdir, trim=True):
    CNOM_srcfile = sorted(glob(f"{srcdir}/input/ECA_EXAA_CPR_NOM_1B_*/ECA_EXAA_CPR_NOM_1B_*.h5"))[-1]
    CNOM = xr.open_dataset(CNOM_srcfile, group='ScienceData/Geo')
    if trim:
        CNOM = trim_to_frame(CNOM, along_track_dim='phony_dim_3', latvar='latitude')
    return CNOM
    
def load_MRGR(srcdir, trim=True):
    MRGR_srcfile = sorted(glob(f"{srcdir}/input/ECA_EXAA_MSI_RGR_1C_*/ECA_EXAA_MSI_RGR_1C_*.h5"))[-1]
    MRGR = xr.open_dataset(MRGR_srcfile, group='ScienceData')
    if trim:
        MRGR = trim_to_frame(MRGR, latvar='latitude', sel_nadir={'across_track':284})
    return MRGR

def load_BNOM(srcdir, trim=True):
    BNOM_srcfile = sorted(glob(f"{srcdir}/input/ECA_EXAA_BBR_NOM_1B_*/ECA_EXAA_BBR_NOM_1B_*.h5"))[-1]
    BNOM = xr.open_dataset(BNOM_srcfile, group='ScienceData/full').sel(view=1)
    if trim:
        BNOM = trim_to_frame(BNOM, latvar='barycentre_latitude')
    return BNOM
    
    
def load_ECL1(srcdir, trim=True):
    ANOM = load_ANOM(srcdir, trim=trim)
    BNOM = load_BNOM(srcdir, trim=trim)
    CNOM = load_CNOM(srcdir, trim=trim)
    MRGR = load_MRGR(srcdir, trim=trim)
    
    return ANOM, CNOM, MRGR, BNOM 
    
def load_ECL2(srcdir, trim=True):
    ATC = load_ATC(srcdir, trim=trim)
    CTC = load_CTC(srcdir, trim=trim)
    ACTC = load_ACTC(srcdir, trim=trim)
    
    AEBD = load_AEBD(srcdir, trim=trim)
    AAER = load_AAER(srcdir, trim=trim)
    AICE = load_AICE(srcdir, trim=trim)
    CCLD = load_CCLD(srcdir, trim=trim)
    ACMCOM = load_ACMCOM(srcdir, trim=trim)
    ACMCAP = load_ACMCAP(srcdir, trim=trim)
    
    return ATC, AEBD, AAER, AICE, CTC, CCLD, ACTC, ACMCAP, ACMCOM