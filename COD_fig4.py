import numpy as np
from netCDF4 import Dataset
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from global_land_mask import globe
con='/home/mhaghigh/nc_file_in_outplume/2sep_omps_cosp.nc'
sat='/home/mhaghigh/nc_file_in_outplume/geo_modis_data_2d.nc'
#a=ipath+re_1sep
nc_m_1=Dataset(con,'r')
tau=nc_m_1.variables['tau_dw'][:,:]
lon_m=nc_m_1.variables['lon'][:]
lat_m=nc_m_1.variables['lat'][:]
nc_m_2=Dataset(sat,'r')
tau_s=nc_m_2.variables['tau'][:,:]
lon=nc_m_2.variables['lon'][:,:]
lat=nc_m_2.variables['lat'][:,:]
tau_s=np.transpose(tau_s)
print(np.amax(tau_s))
lon_mesh,lat_mesh=np.meshgrid(lon_m,lat_m)
globe_land_mask = globe.is_land(lat_mesh,lon_mesh)
tau_o=ma.masked_where(globe_land_mask==True,tau)
tau_s_o=ma.masked_where(globe_land_mask==True,tau_s)
cmap_3='viridis'
cmap = [(0.0,0.0,0.0)] + [(cm.jet(i)) for i in range(1,256)]
cmap = mpl.colors.ListedColormap(cmap)
bounds = np.arange(0,105,5)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cmap_2 = LinearSegmentedColormap.from_list("", ["white","lightskyblue","steelblue","green","yellowgreen","yellow","gold","red","firebrick","darkred"])
fig, (axs0,axs1) = plt.subplots(1,2)
plt.subplot(axs0)
m = Basemap(ax=axs0,projection='cyl',llcrnrlat=50,urcrnrlat=80,llcrnrlon=-60,urcrnrlon=20,resolution='c')
m.drawcoastlines()

axs0=m.imshow(tau_o, cmap=cmap_3,vmin = 0,vmax= 120)
plt.title('ICON',fontsize=20)
plt.subplot(axs1)
m = Basemap(ax=axs1,projection='cyl',llcrnrlat=50,urcrnrlat=80,llcrnrlon=-60,urcrnrlon=20,resolution='c')
m.drawcoastlines()
x, y = m(lon_m,lat_m)
axs1=m.pcolormesh(x, y, tau_s_o, cmap=cmap_3,vmin= 0,vmax= 120)
plt.title('Modis',fontsize=20)

cax = fig.add_axes([0.15,0.25,0.7,0.02])
cbar_bounds = bounds
cbar_ticks = bounds
cbar = fig.colorbar(axs0, cax=cax, norm=norm, boundaries=cbar_bounds, ticks=cbar_ticks, orientation='horizontal',shrink=3)
cbar_labels = str(bounds)
plt.subplots_adjust(left=0.125, bottom=0.3, right=0.9, top=0.7, wspace=0.1, hspace=0.3)
#cbar.set_label('cloud optical thickness',fontsize=20)
#fig.suptitle('2 September 2014', fontsize=20)
cbar.ax.tick_params(labelsize=30)
plt.savefig('paper_figures/tau_fig4.png')
plt.savefig('paper_figures/tau_fig4.pdf')
plt.show()