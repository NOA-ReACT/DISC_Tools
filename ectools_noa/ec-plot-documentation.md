# EarthCARE Plotting Functions Documentation

## Plotting Functions

### Main Plotting Functions
1. `plot_EC_target_classification` - Creates classification plots for EarthCARE target data
2. `plot_EC_2D` - Creates 2D plots for various EarthCARE parameters
3. `plot_scc_2D` - Creates specialized 2D plots for SCC data
4. `plot_EC_1D` - Creates 1D plots for EarthCARE data
5. `plot_RGB_MRGR` - Creates RGB visualization of MRGR data
6. `plot_TIR_MRGR` - Creates thermal infrared visualization of MRGR data
7. `plot_CFAD` - Plots Contoured Frequency by Altitude Diagrams
8. `plot_CFAD_frames` - Creates CFAD plots for different frames
9. `plot_MAOT_example` - Creates example plots for MAOT (Aerosol Optical Thickness) data
10. `plot_MCOP_example` - Creates example plots for MCOP (Cloud Optical Properties) data
11. `plot_MCM_example` - Creates example plots for MCM (Cloud Mask) data

### MSI-Specific Plotting
12. `plot_ECL1_MSI_VNS` - Plots MSI VNS (Visible and Near-infrared) bands
13. `plot_ECL1_MSI_TIR` - Plots MSI TIR (Thermal Infrared) bands
14. `plot_ECL1_MSI_extras` - Plots additional MSI parameters

### Quicklook Functions
15. `quicklook_ANOM` - Creates quicklook plots for ANOM (ATLID Level 1 Nominal) data
16. `quicklook_CNOM` - Creates quicklook plots for CNOM (CPR Level 1 Nominal) data
17. `quicklook_AEBD` - Creates quicklook plots for AEBD data
18. `quicklook_AICE` - Creates quicklook plots for AICE data
19. `quicklook_ATC` - Creates quicklook plots for ATC data
20. `quicklook_CTC` - Creates quicklook plots for CTC data
21. `quicklook_CFMR` - Creates quicklook plots for CFMR data
22. `quicklook_CCD` - Creates quicklook plots for CCD data
23. `quicklook_CCLD` - Creates quicklook plots for CCLD data
24. `quicklook_ACTC` - Creates quicklook plots for ACTC data
25. `quicklook_ACMCAP` - Creates quicklook plots for ACMCAP data
26. `quicklook_MCM` - Creates quicklook plots for MCM data
27. `quicklook_AFM` - Creates quicklook plots for AFM data
28. `quicklook_BSNG` - Creates quicklook plots for BSNG data
29. `quicklook_ACMB` - Creates combined quicklook plots
30. `quicklook_measurements_CNOM` - Creates measurement quicklooks for CNOM data
31. `quicklook_platform_CNOM` - Creates platform quicklooks for CNOM data
32. `quicklook_orbit_CNOM` - Creates orbit quicklooks for CNOM data

### Intercomparison Functions
33. `intercompare_target_classification` - Compares different target classifications
34. `intercompare_ice_water_content` - Compares ice water content measurements
35. `intercompare_snow_rate` - Compares snow rate measurements
36. `intercompare_ice_effective_radius` - Compares ice effective radius measurements
37. `intercompare_rain_water_content` - Compares rain water content measurements
38. `intercompare_rain_rate` - Compares rain rate measurements
39. `intercompare_liquid_water_content` - Compares liquid water content measurements
40. `intercompare_liquid_effective_radius` - Compares liquid effective radius measurements
41. `intercompare_aerosol_extinction` - Compares aerosol extinction measurements
42. `intercompare_lidar_ratio` - Compares lidar ratio measurements
43. `intercompare_rain_median_diameter` - Compares rain median diameter measurements
44. `intercompare_ice_median_diameter` - Compares ice median diameter measurements
45. `intercompare_CFADs` - Compares Contoured Frequency by Altitude Diagrams

## Helper Functions

### Formatting Functions
1. `format_time` - Formats time axis
2. `format_height` - Formats height axis
3. `format_across_track` - Formats across-track axis
4. `format_latitude` - Formats latitude axis
5. `format_plot` - Formats general plot elements
6. `format_plot_1D` - Formats 1D plot elements
7. `format_time_ticks` - Formats time axis ticks
8. `format_latlon_ticks` - Formats latitude/longitude ticks
9. `format_MRGR` - Formats MRGR plots

### Plot Enhancement Functions
10. `add_colorbar` - Adds colorbar to plots
11. `add_nadir_track` - Adds nadir track to plots
12. `add_extras` - Adds extra elements to plots
13. `add_subfigure_labels` - Adds labels to subfigures
14. `add_surface` - Adds surface representation
15. `add_temperature` - Adds temperature contours
16. `add_humidity` - Adds humidity contours
17. `add_land_sea_border` - Adds land-sea border
18. `add_marble` - Adds marble effect to plots
19. `shade_around_text` - Adds shading around text

### Text Processing Functions
20. `linesplit` - Splits text into lines
21. `linebreak` - Breaks text into lines with specified length

### Data Processing Functions
22. `calculate_RGB_MRGR` - Calculates RGB values for MRGR data
23. `create_CFAD` - Creates Contoured Frequency by Altitude Diagrams
24. `stack_histograms` - Stacks histogram data
25. `counts_to_frequency` - Converts counts to frequencies
26. `snap_xlims` - Synchronizes x-axis limits

## Constants and Colormaps

### Colormaps
- `cmap_grey_r` - Reversed grayscale colormap
- `cmap_grey` - Grayscale colormap
- `cmap_rnbw` - Rainbow colormap
- `cmap_org` - Orange colormap
- `cmap_smc` - Seismic colormap

### Category Colors
- `ACTC_category_colors` - Colors for ACTC categories
- `ATC_category_colors` - Colors for ATC categories
- `CTC_category_colors` - Colors for CTC categories
- `MAOT_qstat_category_colors` - Colors for MAOT quality status categories
- `MCOP_qstat_category_colors` - Colors for MCOP quality status categories
- `MCM_maskphase_category_colors` - Colors for MCM mask phase categories
- `MCM_type_category_colors` - Colors for MCM type categories

## Usage Notes

1. Most plotting functions accept parameters for:
   - Dataset containing the data to plot
   - Maximum height (hmax)
   - Plot type specific parameters
   - Output directory for saving plots

2. Helper functions are designed to be used within the plotting functions but can also be used independently.

3. Colormap constants are used consistently across different plot types for visual coherence.

4. Standard plot elements include:
   - Colorbars
   - Axis labels
   - Title
   - Nadir track (where applicable)
   - Surface indicators (where applicable)
