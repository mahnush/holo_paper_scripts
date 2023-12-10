import scipy.interpolate as sci
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
from global_land_mask import globe

nc = Dataset('/home/mhaghigh/ssf_l2/nc_files/CERES_FluxByCldTyp-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20140901-20140930.nc', 'r')
albedo_input = nc.variables['toa_alb_all_daily'][0:5,:,:]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
print(np.shape(lat))
albedo = albedo_input[1, :, :]

cmap = [(0.0,0.0,0.0)] + [(cm.jet(i)) for i in range(1,256)]
cmap = mpl.colors.ListedColormap(cmap)
bounds = np.arange(0,1,0.1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cmap_2 = LinearSegmentedColormap.from_list("", ["lightskyblue","steelblue","green","yellowgreen","yellow","gold","red","firebrick","darkred"])
fig, axs0 = plt.subplots(1,1,figsize=(30,20))
m = Basemap(projection='cyl',llcrnrlat=50,urcrnrlat=80,\
           llcrnrlon=-60,urcrnrlon=20,resolution='c')

m.drawcoastlines()
x, y = m(lon,lat)
axs0.set_title('albedo')
axs0=m.pcolormesh(x, y,albedo_input[1,:,:] ,cmap=cmap_2, norm=norm)

cax = fig.add_axes([0.15,0.1,0.7,0.02])
cbar_bounds = bounds
cbar_ticks =  bounds
cbar = fig.colorbar(axs0,cax=cax, norm=norm, boundaries=cbar_bounds, ticks=cbar_ticks, orientation='horizontal')
cbar.set_label('albedo', fontsize=30)
cbar.ax.tick_params(labelsize='x-large')
plt.show()


