import numpy as np
from netCDF4 import Dataset
from global_land_mask import globe
import numpy.ma as ma
import scipy.interpolate as sci
from global_land_mask import globe
import matplotlib.pyplot as plt



path_s = '/home/mhaghigh/nc_file_in_outplume/'
so2_1sep = path_s + 'so2_Iceland_1sep_0.7res.nc'
so2_2sep = path_s + 'so2_Iceland_2sep_0.7res.nc'
so2_3sep = path_s + 'so2_Iceland_3sep_0.7res.nc'
so2_4sep = path_s + 'so2_Iceland_4sep_0.7res.nc'
so2_5sep = path_s + 'so2_Iceland_5sep_0.7res.nc'


def read_modis(data_name):
    path = '/home/mhaghigh/nc_file_in_outplume/corect_data_model/modis/'
    input_data = path + data_name
    nc = Dataset(input_data,'r')
    cf = nc.variables['clc'][:]
    return cf


def read_model(data_name):
    path = '/home/mhaghigh/nc_file_in_outplume/corect_data_model/'
    input_data = path + data_name
    nc_model = Dataset(input_data,'r')
    cf_model = nc_model.variables['clct_dw'][:]
    return cf_model


def in_out_plume_modis(modis_data,so2_data):
   mdata =  so2_data
   nc_m=Dataset(mdata,'r')
   lat_m=nc_m.variables['lat'][:,:]
   lon_m=nc_m.variables['lon'][:,:]
   so2=nc_m.variables['so2_TRL'][:,:]
   lon_m=lon_m[0:43,0:115]
   lat_m=lat_m[0:43,0:115]
   so2=so2[0:43,0:115]
   so2_mask=np.ma.filled(so2,fill_value=0)
   lon_coarse=lon_m[0,:]
   lat_coarse=lat_m[:,0]
   lat_fine = np.arange(50, 80.05, 0.05)
   lon_fine = np.arange(-60, 20.05, 0.05)
   f = sci.RectBivariateSpline(lat_coarse, lon_coarse,so2_mask )
   scale_interp = f(lat_fine, lon_fine)
   lon_mesh,lat_mesh=np.meshgrid(lon_fine, lat_fine)
   lon_mask_in = ma.masked_where(scale_interp < 1.0, lon_mesh)
   lon_mask_out = ma.masked_where(scale_interp > 1.0, lon_mesh)
   globe_land_mask = globe.is_land(lat_mesh, lon_mesh)
   cf_ocean = ma.masked_where(globe_land_mask == True, read_modis(modis_data))
   cf_in = ma.masked_where(lon_mask_in == True, cf_ocean)
   cf_out = ma.masked_where(lon_mask_out== True, cf_ocean)
   cf_in = cf_in.compressed()
   cf_in = cf_in.flatten()
   cf_out = cf_out.compressed()
   cf_out = cf_out.flatten()
   return cf_in,cf_out


def in_out_plume_model(model_data,so2_data):
   mdata =  so2_data
   nc_m=Dataset(mdata,'r')
   lat_m=nc_m.variables['lat'][:,:]
   lon_m=nc_m.variables['lon'][:,:]
   so2=nc_m.variables['so2_TRL'][:,:]
   lon_m=lon_m[0:43,0:115]
   lat_m=lat_m[0:43,0:115]
   so2=so2[0:43,0:115]
   so2_mask=np.ma.filled(so2,fill_value=0)
   lon_coarse=lon_m[0,:]
   lat_coarse=lat_m[:,0]
   lat_fine = np.arange(50, 80.02, 0.02)
   lon_fine = np.arange(-60, 20.02, 0.02)
   f = sci.RectBivariateSpline(lat_coarse, lon_coarse,so2_mask )
   scale_interp = f(lat_fine, lon_fine)
   lon_mesh,lat_mesh=np.meshgrid(lon_fine, lat_fine)
   lon_mask_in = ma.masked_where(scale_interp < 1.0, lon_mesh)
   lon_mask_out = ma.masked_where(scale_interp > 1.0, lon_mesh)
   globe_land_mask = globe.is_land(lat_mesh, lon_mesh)
   cf_ocean = ma.masked_where(globe_land_mask == True, read_model(model_data))
   cf_in = ma.masked_where(lon_mask_in == True,cf_ocean)
   cf_out = ma.masked_where(lon_mask_out== True,cf_ocean)
   cf_in = cf_in.compressed()
   cf_in = cf_in.flatten()
   cf_out = cf_out.compressed()
   cf_out = cf_out.flatten()
   return cf_in,cf_out

modis_1sep = 'totatl_cloud_fraction_1sep.nc'
modis_2sep = 'totatl_cloud_fraction_2sep.nc'
modis_3sep = 'totatl_cloud_fraction_3sep.nc'
modis_4sep = 'totatl_cloud_fraction_4sep.nc'
modis_5sep = 'totatl_cloud_fraction_5sep.nc'

model_1sep_per  = 'cftm_pert_1sep.nc'
model_2sep_per  = 'cftm_pert_2sep.nc'
model_3sep_per  = 'cftm_pert_3sep.nc'
model_4sep_per  = 'cftm_pert_4sep.nc'
model_5sep_per  = 'cftm_pert_5sep.nc'

model_1sep_con = 'cftm_con_1sep.nc'
model_2sep_con = 'cftm_con_2sep.nc'
model_3sep_con = 'cftm_con_3sep.nc'
model_4sep_con = 'cftm_con_4sep.nc'
model_5sep_con = 'cftm_con_5sep.nc'

control_data = [model_1sep_con,model_2sep_con,model_3sep_con,model_4sep_con,model_5sep_con]
per_data = [model_1sep_per,model_2sep_per,model_3sep_per,model_4sep_per,model_5sep_per]
modis_data = [modis_1sep,modis_2sep,modis_3sep,modis_4sep,modis_5sep]
so2_data = [so2_1sep,so2_2sep,so2_3sep,so2_4sep,so2_5sep]

modis_all_var_in = []
modis_all_var_out = []
for ii in range(5):
    inside,outside = in_out_plume_modis(modis_data[ii],so2_data[ii])
    modis_all_var_in = np.concatenate((modis_all_var_in,inside), axis=0)
    modis_all_var_out= np.concatenate((modis_all_var_out,outside), axis=0)

all_var_con_in = []
all_var_con_out = []
for ii in range(5):
    inside,outside = in_out_plume_model(control_data[ii],so2_data[ii])
    all_var_con_in= np.concatenate((all_var_con_in,inside), axis=0)
    all_var_con_out = np.concatenate((all_var_con_out,outside), axis=0)

all_var_per_in = []
all_var_per_out = []
for ii in range(5):
   inside,outside = in_out_plume_model(per_data[ii],so2_data[ii])
   all_var_per_in= np.concatenate((all_var_per_in,inside), axis=0)
   all_var_per_out = np.concatenate((all_var_per_out,outside), axis=0)

def plt_hist(ax, var, label,color):
   font_legend = 25
   range = np.arange(0, 110,10)
   weight = (np.zeros_like(var) + 1) / len(var)
   mean = str(round(np.ma.mean(var),3))
   median = str(round(np.ma.median(var),4))
   ax.hist(var, bins = range, histtype='step', label=label +' (mean= '+mean+')', density = True,linewidth=4, color = color)
   ax.legend(loc='upper right',fontsize=font_legend)
print('modis_in')
print(np.ma.mean(modis_all_var_in))
print(np.ma.median(modis_all_var_in))
print('modis_out')
print(np.ma.mean(modis_all_var_out))
print(np.ma.median(modis_all_var_out))
print('per_in')
print(np.ma.mean(all_var_per_in))
print(np.ma.median(all_var_per_in))
print('per_out')
print(np.ma.mean(all_var_per_out))
print(np.ma.median(all_var_per_in))
print('con_in')
print(np.ma.mean(all_var_con_in))
print(np.ma.median(all_var_con_in))
print('con_out')
print(np.ma.mean(all_var_con_out))
print(np.ma.median(all_var_con_out))

fig, (axs0,axs1) = plt.subplots(1,2,figsize=(30,20))
plt_hist(axs0, modis_all_var_in, 'modis','black')
plt_hist(axs1, modis_all_var_out,'modis','black')
plt_hist(axs0, all_var_con_in,'control','red')
plt_hist(axs1, all_var_con_out,'control','red')
plt_hist(axs0, all_var_per_in,'perturbed','blue')
plt_hist(axs1, all_var_per_out,'perturbed','blue')
plt.show()