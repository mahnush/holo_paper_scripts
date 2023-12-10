import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


def read_data(data_name):
    input_path = '/home/mhaghigh/nc_file_in_outplume/'
    input_name = input_path + data_name
    input_cams = Dataset(input_name, 'r')
    sulfat_con = input_cams.variables['so2_TRL'][:, :]
    sulfat_con_test = sulfat_con[sulfat_con < 1.0]
    sulfat_con_test_mean = np.ma.mean(sulfat_con_test)
    print(sulfat_con_test_mean)
    lat = input_cams.variables['lat'][:,:]
    lon = input_cams.variables['lon'][:,:]
    so2 = np.ma.masked_where(sulfat_con<1,sulfat_con)
    return so2, lat, lon


def visualize(row, j, data_name,day):
    sulfat_mean, lat, lon = read_data(data_name)
    cmap = [(0.0, 0.0, 0.0)] + [(cm.jet(i)) for i in range(1, 256)]
    cmap = mpl.colors.ListedColormap(cmap)
    bounds = np.arange(0, 22, 2)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap_2 = LinearSegmentedColormap.from_list("", ["white","lightskyblue", "steelblue", "green", "yellowgreen",
                                                    "yellow", "gold", "red", "firebrick", "darkred"])
    m = Basemap(ax=axs[row, j], projection='cyl', llcrnrlat=50, urcrnrlat=80, llcrnrlon=-60,
                urcrnrlon=20, resolution='c')
    #m.drawmeridians(np.arange(-180., 180.,10.), linewidth=1.2, labels=[0, 0, 0, 1], fontsize = 20, color='grey', zorder=3, latmax=90,
    #             )
    #m.drawparallels(np.arange(0., 85., 10.), linewidth=1.2, labels=[1, 0, 0, 0],  fontsize = 20, color='grey', zorder=3, latmax=90,
    #              )
    m.drawcoastlines()
    title = str(day)+' Sep '
    axs[row, j].set_title(title, fontsize=60)
    axs[row, j] = m.imshow(sulfat_mean, cmap=cmap_2, norm=norm)
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.02])   #left, bottom, width,height
    cbar_bounds = bounds
    cbar_ticks = cbar_bounds
    cbar = fig.colorbar(axs[row, j], cax=cax, norm=norm, boundaries=cbar_bounds,
                        ticks=cbar_ticks, orientation='horizontal')
    cbar.ax.tick_params(labelsize=60)
    cbar.set_label('Vertical column of SO2 (DU)', fontsize=50)

    return
fig, axs = plt.subplots(3, 2, figsize=(30, 20),sharex = True, sharey = True)
#plt.tight_layout()
visualize(0, 0, 'so2_Iceland_1sep_0.7res.nc', 1)
visualize(0, 1,'so2_Iceland_2sep_0.7res.nc', 2)
visualize(1, 0, 'so2_Iceland_3sep_0.7res.nc', 3)
visualize(1, 1, 'so2_Iceland_4sep_0.7res.nc',4)
visualize(2, 0, 'so2_Iceland_5sep_0.7res.nc',5)
visualize(2, 1, 'so2_Iceland_7sep_0.7res.nc',7)

plt.savefig('revised_paper_output/so2_concentration_fig2.png')
plt.savefig('so2_concentration_fig2_poster.png',dpi = 500)
plt.show()