from pyhdf.SD import SD, SDC
import numpy as np
from netCDF4 import Dataset
import MODIS_functions
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.cm as cm
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import numpy.ma as ma


limit = np.zeros(4)
#limit[0] = 6
#limit[1] = 34
limit[0] = 50
limit[1] = 80
limit[2] = -60
limit[3] = 20
gridSize = 0.2

nc = Dataset('/home/mhaghigh/ssf_l2/nc_files/CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014090100-2014090123.nc', 'r')
#read in the variables of choice
sw_flux_inward_input = nc.variables['TOA_Incoming_Solar_Radiation'][:]
sw_flux_upward_input = nc.variables['CERES_SW_TOA_flux___upwards'][:]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

sw_flux_upward_input_grid = MODIS_functions.grid(limit, gridSize, sw_flux_upward_input, lat, lon)
sw_flux_inward_input_grid = MODIS_functions.grid(limit, gridSize, sw_flux_inward_input, lat, lon)
sw_flux_upward_input_grid = np.transpose(sw_flux_upward_input_grid)
sw_flux_inward_input_grid = np.transpose(sw_flux_inward_input_grid)
print(np.shape(sw_flux_upward_input_grid))


#to plot
fig = plt.figure(figsize=(10, 8))
x = np.arange(-60, 20.2, 0.2)
y = np.arange(50, 80.2, 0.2)
xv, yv = np.meshgrid(x, y)
print(np.shape(xv))
print(np.shape(yv))
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
m.pcolormesh(xv, yv, sw_flux_upward_input_grid)
plt.colorbar()
plt.show()

ncout = Dataset('/home/mhaghigh/nc_file_in_outplume/test_ceres.nc',mode="w",format='NETCDF4_CLASSIC')
nlon = 401
nlat = 151


ncout.createDimension('lon', nlon)
ncout.createDimension('lat', nlat)
lon_o = ncout.createVariable('lon',np.float32,('lon',))
lat_o= ncout.createVariable('lat',np.float32,('lat',))
down_mean_o = ncout.createVariable('down_flux',np.float32,('lat','lon'))

print(np.shape(x))
lon_o[:] = x[:]
lat_o[:] = y[:]
down_mean_o[:] = sw_flux_upward_input_grid[:]
