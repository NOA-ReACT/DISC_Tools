import cpr

# C-NOM
cpr_file = 'ECA_EXAA_CPR_NOM_1B_20241231T183449Z_20231012T115403Z_39316D.h5'
group = 'ScienceData/Data'
var_name = 'radarReflectivityFactor'
png_file = f'{var_name}.png'
cpr.plot(cpr_file,group,var_name,png_file)

cpr_file = 'ECA_EXAA_CPR_NOM_1B_20241231T183449Z_20231012T115403Z_39316D.h5'
group = 'ScienceData/Data'
var_name = 'dopplerVelocity'
png_file = f'{var_name}.png'
cpr.plot(cpr_file,group,var_name,png_file)

# # C-FMR
cpr_file = 'ECA_EXAA_CPR_FMR_2A_20241231T183449Z_20240327T192758Z_39316D.h5'
group = 'ScienceData'
var_name = 'reflectivity_corrected'
png_file = f'{var_name}.png'
cpr.plot(cpr_file,group,var_name,png_file)

# # C-CD
cpr_file = 'ECA_EXAA_CPR_CD__2A_20241231T183449Z_20240327T192758Z_39316D.h5'
group = 'ScienceData'
var_name = 'doppler_velocity_integrated'
png_file = f'{var_name}.png'
cpr.plot(cpr_file,group,var_name,png_file)

# C-TC
cpr_file = 'ECA_EXAA_CPR_TC__2A_20241231T183449Z_20240327T192758Z_39316D.h5'
group = 'ScienceData'
var_name = 'hydrometeor_classification'
png_file = f'{var_name}.png'
cpr.plot(cpr_file,group,var_name,png_file)

# C-CLD
cld_file = 'ECA_EXAA_CPR_CLD_2A_20241231T183449Z_20240326T174030Z_39316D.h5'
group = 'ScienceData'
var_name = 'water_content'
png_file = f'{var_name}.png'
cpr.plot(cld_file,group,var_name,png_file)

var_name = 'characteristic_diameter'
png_file = f'{var_name}.png'
cpr.plot(cld_file,group,var_name,png_file)