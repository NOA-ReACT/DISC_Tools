# EarthCARE Statistical Validation Tools

Statistical analysis package for comparing EarthCARE satellite measurements with ground-based lidar observations.

## Overview

This package provides a complete workflow for atmospheric profile validation, from data loading through statistical analysis and visualization. It supports both profile statistics (altitude-dependent bias analysis) and scatter plot analysis (overall correlation assessment).

## Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `statconfig.py` | Configuration parameters | `DEFAULT_CONFIG_ST`, validation functions |
| `statio.py` | Data I/O and preprocessing | `load_all_events()`, `data_preparation_single_event()` |
| `stat_calc.py` | Statistical calculations | `calculate_bias()`, `profile_statistics()`, `calculate_scatter_stats()` |
| `statplot.py` | Visualization functions | `create_profile_figure()`, `create_scatter_plots()` |
| `statistics_core_v1.py` | Main orchestration script | `main()` workflow |

## Quick Start

1. **Configure analysis parameters** in `statconfig.py`:
   ```python
   DEFAULT_CONFIG_ST = {
       'EVENT_LIST': ['/path/to/event1', '/path/to/event2'],
       'NETWORK': 'POLLYXT',
       'MAX_DISTANCE': 50,  # km
       'VARIABLES': ['particle_backscatter_coefficient_355nm', ...]
   }
   ```

2. **Run complete analysis**:
   ```bash
   python statistics_core_v1.py
   ```

## Analysis Workflow

```
Load Events → Calculate Statistics → Create Plots
     ↓              ↓                    ↓
  statio.py    stat_calc.py         statplot.py
```

### Profile Analysis
- Computes bias (satellite - ground) at different altitudes
- Shows where retrieval errors occur in atmospheric layers
- Output: Vertical bias profiles with error bars

### Scatter Analysis  
- Evaluates overall correlation between measurements
- Calculates regression statistics (slope, intercept, R, RMSE)
- Output: Ground vs satellite scatter plots with fit lines

## Key Features

- **Consistent data masking** between profile and scatter workflows
- **Cloud filtering** using classification variables
- **Outlier detection** with IQR or Z-score methods
- **Height binning** for profile statistics
- **Automated plotting** with publication-ready figures

## Configuration Options

| Parameter | Description | Options |
|-----------|-------------|---------|
| `GROUPING_METHOD` | Height binning approach | `'height'`, `'km_bins'` |
| `ENABLE_FILTERING` | Apply cloud filtering | `True`, `False` |
| `REMOVE_OUTLIERS` | Remove statistical outliers | `True`, `False` |
| `CERTAIN_PROFILES` | Use closest satellite profiles only | `True`, `False` |

## Output Files

- Profile plots: `profile_stats_mean_[distance]_[method].png`
- Scatter plots: `scatter_stats_mean_[distance]_[method].png`
- Statistics saved in xarray datasets for further analysis

## Dependencies

- **Core**: numpy, pandas, xarray, matplotlib, scipy
- **EarthCARE tools**: ectools_noa, valio, valplot
- **Optional**: cartopy (for geographic projections)

## Author

Andreas Karipis - National Observatory of Athens (NOA), ReACT
