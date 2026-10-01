#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network registry for the valtools.

One entry per ground-based network: which ground variables to plot and their
ranges, how to read the ground times, the figure title and how to read the
ground profiles. valtool_manager looks the network up here instead of using
if/elif branches.

Keys per network
----------------
crop_quicklook : bool  | Crop the GND quicklook around the overpass (crop_polly_file)
scc_term       : str   | SCC product term passed to load_ground_data (EARLINET)
quicklook      : dict  | Per level ('L1', 'L2'): variables, titles, scales,
                          ranges, units, heightvar of the two GND quicklooks.
                          Titles are prefixed with the station name at plot time.
time           : dict  | Ground profile time variables:
                          start : variable holding the (start) time
                          end   : variable holding the end time, None if the
                                  profile has a single time (no averaging window).
                                  If the variable is missing in a file, no window.
                          unit  : unit of the stored value for pd.to_datetime
                                  ('s', 'ns', ...), None if already a datetime
suptitle       : str   | L2 figure title template, None for no title. Fields:
                          {b0} {b1} {overpass_time} {station_name} {keyword} {gnd_time}
use_retrieval  : bool  | Apply the RETRIEVAL config (Raman/Klett selection)
profiles       : func  | profiles(gnd_profile) -> yields (raman, klett) per
                          ground profile; either can be None

Private networks
----------------
Networks kept out of the public release go in val_networks_local.py as
NETWORKS_LOCAL = {...} (same structure). They are merged here if the file exists.
"""
import pandas as pd

from valio import read_pollynet_profile, read_scc_profile


# ---------------------------------------------------------------- readers
def _pollyxt_profiles(gnd_profile):
    """PollyNET: one Dataset with a 'time' dimension -> (raman, klett) per time."""
    for t in range(gnd_profile.sizes['time']):
        yield read_pollynet_profile(gnd_profile.isel(time=t), data=True)


def _earlinet_profiles(gnd_profile):
    """EARLINET/SCC: list of Datasets sharing a 'time' dimension -> (raman, klett) per time."""
    for t in range(gnd_profile[0].sizes['time']):
        yield read_scc_profile(gnd_profile, t)


# ---------------------------------------------------------------- registry
_STD_SUPTITLE = ('EarthCARE A-EBD({b0}) & A-TC({b1}) Comparison at {overpass_time} UTC with\n'
                 ' {station_name} Ground Station L2 {keyword} Retrieval at {gnd_time} UTC')

NETWORKS = {
    'POLLYXT': {
        'crop_quicklook': True,
        'quicklook': {
            'L1': dict(variables=['attenuated_backscatter_355nm', 'volume_depolarization_ratio_355nm'],
                       titles=['att.bsc', 'vol.depol.ratio'],
                       scales=['log', 'linear'],
                       ranges=[[1e-8, 3e-5], [0.0, 0.3]],
                       units=['m⁻¹ sr⁻¹', '-'],
                       heightvar='height'),
            'L2': dict(variables=['quasi_bsc_532', 'quasi_pardepol_532'],
                       titles=['att.bsc', 'par.depol.ratio'],
                       scales=['log', 'linear'],
                       ranges=[[1e-8, 15e-6], [0, 0.4]],
                       units=['m⁻¹ sr⁻¹', '-'],
                       heightvar='height'),
        },
        'time': dict(start='start_time', end='end_time', unit='s'),
        'suptitle': _STD_SUPTITLE,
        'use_retrieval': True,
        'profiles': _pollyxt_profiles,
    },

    'EARLINET': {
        'crop_quicklook': True,
        'scc_term': 'b0355',
        'quicklook': {
            'L1': dict(variables=['range_corrected_signal', 'volume_linear_depolarization_ratio'],
                       titles=['range.cor.signal', 'vol.depol.ratio'],
                       scales=['log', 'linear'],
                       ranges=[[1e7, 1e9], [0.0, 0.2]],
                       units=['m⁻¹ sr⁻¹', '-'],
                       heightvar='altitude'),
            'L2': dict(variables=['range_corrected_signal', 'volume_linear_depolarization_ratio'],
                       titles=['range.cor.signal', 'vol.depol.ratio'],
                       scales=['log', 'linear'],
                       ranges=[[1e7, 1e9], [0.0, 0.4]],
                       units=['m⁻¹ sr⁻¹', '-'],
                       heightvar='altitude'),
        },
        'time': dict(start='start_time', end='end_time', unit='ns'),   # from time_bounds
        'suptitle': _STD_SUPTITLE,
        'use_retrieval': False,        # plots whatever the SCC product provides
        'profiles': _earlinet_profiles,
    },
}

# merge private networks, if available
try:
    from val_networks_local import NETWORKS_LOCAL
    NETWORKS.update(NETWORKS_LOCAL)
except ImportError:
    pass


# ---------------------------------------------------------------- helpers
def get_network(network):
    """Registry entry of a network; clear error if it is not defined."""
    if network not in NETWORKS:
        raise ValueError(f"Unsupported network: {network}. "
                         f"Available: {', '.join(NETWORKS)}")
    return NETWORKS[network]


def get_quicklook(network, level):
    """GND quicklook settings of a network for level 'L1' or 'L2'."""
    ql = get_network(network)['quicklook'].get(level)
    if ql is None:
        raise ValueError(f"No {level} quicklook settings for network: {network}")
    return ql


def get_gnd_times(gnd_profile, time_cfg):
    """(start, end) of a ground profile as pd.Timestamp; end is None if not defined."""
    def _read(var):
        value = gnd_profile[var].values.item()
        if time_cfg['unit']:
            t = pd.to_datetime(value, unit=time_cfg['unit'])
        else:
            t = pd.to_datetime(value)
        # stored times can be off by a few ns (e.g. 23:37:59.999999) -> round to the second
        return t.round('s')

    t0 = _read(time_cfg['start'])

    end_var = time_cfg['end']
    if end_var is not None and end_var in gnd_profile:
        t1 = _read(end_var)
    else:
        t1 = None
    return t0, t1