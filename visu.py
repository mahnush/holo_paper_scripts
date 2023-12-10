import matplotlib.pyplot as plt
import numpy as np



def read_and_visu(netcdf_data,variable_name, row,j, day, bounds):
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from netCDF4 import Dataset
    from mpl_toolkits.basemap import Basemap
    from global_land_mask import globe
    import numpy.ma as ma


    nc = Dataset(netcdf_data)
    variable = nc.variables[variable_name][:]
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    lon_mesh,lat_mesh=np.meshgrid(lon,lat)
    globe_land_mask = globe.is_land(lat_mesh,lon_mesh)
    variable_ocean = ma.masked_where(globe_land_mask==True,variable)
    cmap = [(0.0, 0.0, 0.0)] + [(cm.jet(i)) for i in range(1, 256)]
    cmap = mpl.colors.ListedColormap(cmap)
    #interval =(np.nanmax(variable)-np.nanmin(variable))/10
    #bounds = np.arange(np.nanmin(variable),np.nanmax(variable))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap_2 = LinearSegmentedColormap.from_list("", ["white","lightskyblue", "steelblue", "green", "yellowgreen",
                                                    "yellow", "gold", "red", "firebrick", "darkred"])

    m = Basemap(ax=axs[row, j], projection='cyl', llcrnrlat=50, urcrnrlat=80, llcrnrlon=-60,
                urcrnrlon=20, resolution='c')
    m.drawcoastlines()

    axs[row, j].set_title(str(day) +' sep 2014', fontsize=30)
    axs[row, j] = m.imshow(variable, cmap=cmap_2, norm=norm)
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar_bounds = bounds
    cbar_ticks = cbar_bounds
    cbar = fig.colorbar(axs[row, j], cax=cax, norm=norm, boundaries=cbar_bounds,
                        ticks=cbar_ticks, orientation='horizontal')
    #cbar.set_label(day, fontsize=20)
    cbar.ax.tick_params(labelsize='large')
    return


data_name_1sep = '/home/mhaghigh/nc_file_in_outplume/flux_con_1sep_20km_noon.nc'
data_name_2sep = '/home/mhaghigh/nc_file_in_outplume/flux_con_2sep_20km_noon.nc'
data_name_3sep = '/home/mhaghigh/nc_file_in_outplume/flux_con_3sep_20km_noon.nc'
data_name_4sep = '/home/mhaghigh/nc_file_in_outplume/flux_con_4sep_20km_noon.nc'
data_name_5sep = '/home/mhaghigh/nc_file_in_outplume/flux_con_5sep_20km_noon.nc'
data_name_6sep = '/home/mhaghigh/nc_file_in_outplume/flux_con_6sep_20km_noon.nc'
fig, axs = plt.subplots(3, 2, figsize=(30, 20))
name = 'down_flux'
bounds = np.arange(0,1000,100)
read_and_visu(data_name_1sep,name,0,0,1,bounds)
read_and_visu(data_name_2sep,name,0,1,2,bounds)
read_and_visu(data_name_3sep,name,1,0,3,bounds)
read_and_visu(data_name_4sep,name,1,1,4,bounds)
read_and_visu(data_name_5sep,name,2,0,5,bounds)
read_and_visu(data_name_6sep,name,2,1,6,bounds)
plt.savefig('incoming_solar.pdf')
plt.show()