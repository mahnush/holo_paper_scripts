import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.cm as cm
import numpy.ma as ma
import matplotlib.pyplot as plt


nc = Dataset('/home/mhaghigh/ssf_l2/nc_files/CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014090100-2014090123.nc', 'r')
#read in the variables of choice
sw_flux_inward_input = nc.variables['TOA_Incoming_Solar_Radiation'][:]
sw_flux_upward_input = nc.variables['CERES_SW_TOA_flux___upwards'][:]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

bins = {}
#set the domain that you want to remap and the resolousion
#here is 20km
bins['lat'] = np.arange(50, 81, 0.2)
bins['lon'] = np.arange(-60, 21, 0.2)
#set the non defined values to 0
sw_flux_inward_input = ma.filled(sw_flux_inward_input, fill_value=0)
sw_flux_upward_input = ma.filled(sw_flux_upward_input, fill_value=0)

#computing albedo at top of atmosphare
alb = sw_flux_upward_input/sw_flux_inward_input

#here is the part remapping take place. it can be used for any variable of choice
albedo = np.histogram2d(lon, lat, bins=[bins['lon'], bins['lat']], weights=alb)[0]
income_flux = np.histogram2d(lon, lat, bins=[bins['lon'], bins['lat']], weights=sw_flux_inward_input)[0]
upward_flux = np.histogram2d(lon, lat, bins=[bins['lon'], bins['lat']], weights=sw_flux_upward_input)[0]
b = np.histogram2d(lon, lat,  bins=[bins['lon'], bins['lat']])[0]

#to flip and rotate 90 degree the values I can not remember why it was necessary
var_albedo = np.fliplr(np.rot90(albedo/b, -1))
var_up_sw = np.fliplr(np.rot90(upward_flux/b, -1))
var_down_sw = np.fliplr(np.rot90(income_flux/b, -1))

#here mask the invalid values of albedo
albedo = ma.masked_invalid(albedo)

#this part plot a geographical distributio of variable of choice here is albedo\\
# this part in not necessary for you
fig = plt.figure(figsize=(10, 8))
x = np.arange(-60, 21, 0.2)
y = np.arange(50, 81, 0.2)
xv, yv = np.meshgrid(x, y)
bounds = np.arange(0,1000)
cmap = [(0.0, 0.0, 0.0)] + [(cm.jet(i)) for i in range(1, 256)]
cmap = mpl.colors.ListedColormap(cmap)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cmap_2 = LinearSegmentedColormap.from_list("", ["white","lightskyblue", "steelblue", "green", "yellowgreen",
                                                    "yellow", "gold", "red", "firebrick", "darkred"])
m = Basemap(projection='cyl', resolution='c', \
            llcrnrlat=50, urcrnrlat=80, \
            llcrnrlon=-60, urcrnrlon=20)
#m.pcolormesh(xv, yv,var_albedo, cmap=cmap_2)
m.pcolormesh(xv, yv,var_down_sw, cmap=cmap_2)
plt.colorbar(cmap = cmap_2)
m.drawcoastlines();

#to plot a histigram of the variable here is albedo\\this part is not necessary for you
def plt_hist(var,label):
    var_f = var.flatten()
    var_f = ma.masked_invalid(var_f)
    var_f = ma.compressed(var_f)
    range = (0.001, 1)
    weight = (np.zeros(len(var_f)) + 1) / len(var_f)
    mean_data = np.ma.mean(var_f)
    mean_data = str(round(mean_data,4))
    ax0.hist(var_f, bins=50, range=range, weights=weight, histtype='step', label=label+ ' (mean = '+mean_data+')' , linewidth=2)
    ax0.legend()
fig, ax0 = plt.subplots()
plt_hist(var_albedo, 'CERES')

#to write the data in netcdf file
ncout = Dataset('/home/mhaghigh/nc_file_in_outplume/check_ceres_remap_new_methode_1sep_mask.nc',mode="w",format='NETCDF4_CLASSIC')
nlon = 404
nlat = 154
x =x[0:404]
y = y[0:154]

ncout.createDimension('lon', nlon)
ncout.createDimension('lat', nlat)
lon_o = ncout.createVariable('lon',np.float32,('lon',))
lat_o= ncout.createVariable('lat',np.float32,('lat',))
down_mean_o = ncout.createVariable('down_flux',np.float32,('lat','lon'))
up_mean_o= ncout.createVariable('up_flux',np.float32,('lat','lon'))
alb_mean_o= ncout.createVariable('alb',np.float32,('lat','lon'))
print(np.shape(x))
lon_o[:] = x[:]
lat_o[:] = y[:]
down_mean_o[:] = var_down_sw[:]
up_mean_o[:] = var_up_sw[:]
alb_mean_o[:] = var_albedo[:]


plt.savefig('albedo_regrid.png')
plt.show()
