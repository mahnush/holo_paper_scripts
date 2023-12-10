from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from global_land_mask import globe
nc = Dataset('/home/mhaghigh/ssf_l2/CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014090500-2014090518.nc')
sw_in = nc.variables['TOA_Incoming_Solar_Radiation'][:]
sw = nc.variables['CERES_SW_TOA_flux___upwards'][:]
tau = nc.variables['Mean_visible_optical_depth_for_cloud_layer'][:]
tem = nc.variables['Mean_cloud_top_temperature_for_cloud_layer'][:]
cf_test =  nc.variables['Clear_layer_overlap_percent_coverages'][:]
print(np.shape(cf_test))
cf = cf_test[:,1]
print(np.min(cf))
tem_test = tem[:,0]
tau_test = tau[:,0]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
bins = {}
bins['lat'] = np.arange(50, 81, 0.2)
bins['lon'] = np.arange(-60, 21, 0.2)
lat_a = np.arange(50, 81, 0.2)
lon_a = np.arange(-60, 21, 0.2)
lat_a = lat_a[0:154]
lon_a = lon_a[0:404]
sw[tau_test<4] = 0
#sw[tau_test==np.nan] = 0
sw[tem_test<235] = 0
sw[cf == 0] = 0
#sw_mask = ma.masked_where(sw == 0, sw)
#alb = sw_mask/sw_in
sw_flux_inward_input = ma.filled(sw_in,fill_value=0)
sw_flux_upward_input = ma.filled(sw,fill_value=0)
alb = sw_flux_upward_input/sw_flux_inward_input
alb =  ma.masked_where(alb==0 , alb)
#test = ma.masked_equal(alb,0,copy=True)
albedo = np.histogram2d(lon, lat, bins=[bins['lon'], bins['lat']], weights=alb)[0]
income_flux = np.histogram2d(lon, lat, bins=[bins['lon'], bins['lat']], weights=sw_flux_inward_input)[0]
upward_flux = np.histogram2d(lon, lat, bins=[bins['lon'], bins['lat']], weights=sw_flux_upward_input)[0]
#c = np.histogram2d(lon, lat, bins=[bins['lon'], bins['lat']], weights=np.cos(np.deg2rad(lat))*111*111)[0]
b = np.histogram2d(lon, lat,  bins=[bins['lon'], bins['lat']])[0]
var_albedo = np.fliplr(np.rot90(albedo/b, -1))
var_albedo =  ma.masked_where(var_albedo==0 , var_albedo)
var_albedo  =  ma.masked_where(var_albedo==np.nan,var_albedo)
lon_mesh,lat_mesh=np.meshgrid(lon_a,lat_a)
globe_land_mask = globe.is_land(lat_mesh,lon_mesh)
alb_ocean = ma.masked_where(globe_land_mask==True, var_albedo)
print(np.nanmin(var_albedo))
def plt_hist(var,label):
    var_f = var.flatten()
    var_f = ma.masked_invalid(var_f)
    var_f = ma.compressed(var_f)
    range = np.arange(0, 1,0.01)
    weight = (np.zeros(len(var_f)) + 1) / len(var_f)
    mean_data = np.ma.mean(var_f)
    mean_data = str(round(mean_data,4))
    ax0.hist(var_f, bins=range, weights=weight, histtype='step', label=label+ ' (mean = '+mean_data+')' , linewidth=2)
    ax0.legend()
fig, ax0 = plt.subplots()
plt_hist(alb_ocean, 'CERES')
plt.show()