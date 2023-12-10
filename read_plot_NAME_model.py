import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

path = '/home/mhaghigh/name_model/'


def read_daily_data(day):
    input_path = path + day +'/'
    print(input_path)
    file_list = open(input_path +'file_list.txt')
    k = 0
    sulfate_concentration = np.zeros((24, 22, 150, 200))
    for file_name in file_list:
        file_name = file_name.strip()
        file_name = input_path+file_name
        input_name_model = Dataset(file_name, 'r')
        sulfate_concentration[k, :, :, :] = input_name_model.variables['sulphur-dioxide_air_concentration'][:, 100:250, 300:500]*0.001
        lon = input_name_model.variables['longitude'][300:500]
        lat = input_name_model.variables['latitude'][100:250]
        k = k+1
    time_mean = np.mean(sulfate_concentration, axis=0)
    height_mean = np.mean(time_mean, axis=0)
    return height_mean

def plo_daily(row, j, day):
    sulfate_con = read_daily_data(day)
    print(np.shape(sulfate_con))
    print(np.amax(sulfate_con))
    cmap = [(0.0, 0.0, 0.0)] + [(cm.jet(i)) for i in range(1, 256)]
    cmap = mpl.colors.ListedColormap(cmap)
    bounds = np.arange(0, 1e-7, 5e-9)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap_2 = LinearSegmentedColormap.from_list("", ["white","lightskyblue", "steelblue", "green", "yellowgreen",
                                                    "yellow", "gold", "red", "firebrick", "darkred"])

    m = Basemap(ax=axs[row, j], projection='cyl', llcrnrlat=50, urcrnrlat=80, llcrnrlon=-60,
                urcrnrlon=20, resolution='c')
    m.drawcoastlines()
    axs[row, j].set_title(day, fontsize=30)
    axs[row, j] = m.imshow(sulfate_con, cmap=cmap_2, norm=norm)
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar_bounds = bounds
    cbar_ticks = cbar_bounds
    cbar = fig.colorbar(axs[row, j], cax=cax, norm=norm, boundaries=cbar_bounds,
                        ticks=cbar_ticks, orientation='horizontal')
    #cbar.set_label(day, fontsize=20)
    cbar.ax.tick_params(labelsize='large')
    return


fig, axs = plt.subplots(3, 2, figsize=(30, 20))
plo_daily(0, 0, '1sep')
plo_daily(0, 1, '2sep')
plo_daily(1, 0, '3sep')
plo_daily(1, 1, '4sep')
plo_daily(2, 0, '5sep')
#plo_daily(2, 1, '6sep')
plo_daily(2, 1, '7sep')
plt.savefig('sulfate_aerosol_NAME_model.png')
plt.show()