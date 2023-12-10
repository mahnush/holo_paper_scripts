import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


def read_data(day):
    i=(day-1)*8
    j=day*8
    input_name = '/home/mhaghigh/1st_paper/ncfile/omps_holo_ml.nc'
    input_cams = Dataset(input_name, 'r')
    sulfat_con = input_cams.variables['aermr11'][i:j, :, :, :]
    time_mean = np.ma.mean(sulfat_con, axis=0)
    height_mean = np.ma.mean(time_mean, axis=0)
    print(np.amax(height_mean))
    return height_mean


def visualize(row, j, day):
    sulfat_mean = read_data(day)
    cmap = [(0.0, 0.0, 0.0)] + [(cm.jet(i)) for i in range(1, 256)]
    cmap = mpl.colors.ListedColormap(cmap)
    bounds = np.arange(0, 1e-7, 5e-9)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap_2 = LinearSegmentedColormap.from_list("", ["white", "lightskyblue", "steelblue", "green", "yellowgreen",
                                                    "yellow", "gold", "red", "firebrick", "darkred"])
    m = Basemap(ax=axs[row, j], projection='cyl', llcrnrlat=50, urcrnrlat=80, llcrnrlon=-60,
                urcrnrlon=20, resolution='c')
    m.drawcoastlines()
    title = str(day)+' Sep'
    axs[row, j].set_title(title, fontsize=30)
    axs[row, j] = m.imshow(sulfat_mean, cmap=cmap_2, norm=norm)
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])   #left, bottom, width,height
    cbar_bounds = bounds
    cbar_ticks = cbar_bounds
    cbar = fig.colorbar(axs[row, j], cax=cax, norm=norm, boundaries=cbar_bounds,
                        ticks=cbar_ticks, orientation='horizontal')
    cbar.ax.tick_params(labelsize='xx-large')
    #cbar.set_label('number of activatd CCN (cm-3)', fontsize=20)

    return
fig, axs = plt.subplots(3, 2, figsize=(30, 20))
visualize(0, 0, 1)
visualize(0, 1, 2)
visualize(1, 0, 3)
visualize(1, 1, 4)
visualize(2, 0, 5)
visualize(2, 1, 7)
plt.savefig('/home/mhaghigh/1st_paper/sulfate_aerosol_cams.png')
plt.show()