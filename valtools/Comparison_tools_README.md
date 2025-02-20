# EarthCARE Data Visualization Comparison Tools

Tools for creating comparison plots between EarthCARE ATLID data and ground-based measurements.

## Components

### L1 Tool (`L1_tool.py`)
- **Purpose:** Creates comparison plots between EarthCARE L1 ATLID data, simulator data, and ground measurements
- **Required Data:**
  - ANOM from EC user
  - Simulator file
  - POLLYXT data: att_bsc.nc & vol_depol.nc files
  - EARLINET data: Hirelpp files
- **Logic Flow:** 
  1. Get EC_data, sim_data, gnd_data
  2. Crop data around station
  3. Plot quicklooks, profiles, maps
- **Outputs:**A figure containing:
  - Three L1 A-NOM product quicklooks
  - Two ground station quicklooks
  - Three profile comparisons
  - Geographical context map

### L2 Tool (`L2_tool.py`)
- **Purpose:** Creates comparison plots between EarthCARE L2 ATLID data and ground-based measurements
- **Required Data:**
  - AEBD & ATC from EC
  - profile.nc from POLLYXT
- **Logic Flow:**
  1. Get EC_data, gnd_data
  2. Crop data around station
  3. Find closest overpass point
  4. Separate Raman & Klett parameters
  5. For each point in the 5 closest:
     - Process Raman and Klett data
     - Generate quicklooks, profiles, maps
- **Outputs:**
  - Five L2 products quicklooks
  - Two ground station quicklooks
  - Four profile comparisons
  - Classification and quality status plot
  - Geographical map

## Dependencies
- Python 3.x
- Required packages: pandas, xarray, matplotlib, numpy, geopy, cartopy, seaborn

## Station Networks
Supports data from:
- EARLINET for L1_tool
- POLLYXT for both L1_tool & L2_tool

## Running the Tools

1. **Configure Settings:**
   - Set root directory - (`L*_tool.py`)
   - Adjust default configuration: - (`valtool_manager.py`)
     - Ground Network
     - Profile scale (log/linear)
     - Variables for plotting
     - Axes limits

2. **Execute:**
   ```python
   python L1_tool.py
   python L2_tool.py
   ```

## Supporting Modules
- `valtool_manager.py`:  Main plotting functions and default configurations for
                         L1 and L2 tools
- `valio.py`: Data processing functions
- `valplot.py`: Plotting functions
- `val_L2_dictionaries.py`: Classification dictionaries
- `ecio.py` & `ecplot.py`: From ectools
Note: Plotting functions from `valplot.py` can be used independently for standalone 
plotting tasks during Cal/Val activities.

## Input Data Organization
Copyroot_directory/
├── L1/
│   ├── eca/
│   │   └── *ATL_NOM*.h5
│   ├── sim/
│   │   └── *ATL_NOM*.h5
│   └── gnd/
│       ├── scc/
│       └── tropos/
├── L2/
│   ├── eca/
│   │   ├── *ATL_EBD*.h5
│   │   └── *ATL_TC__.h5
│   └── gnd/
│       ├── scc/
│       └── tropos/
└── plots_comparison/

## Author
Andreas Karipis - NOA - a.karipis@noa.gr