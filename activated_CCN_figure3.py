import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma


file_per = '/home/mhaghigh/1st_paper/ncfile/per_cams_boxmodel.nc'
input_name_perturbed = Dataset(file_per, 'r')
file_control = '/home/mhaghigh/1st_paper/ncfile/holo_test_fig3.nc'

nlat=43
nlon=114
nlev=60
nt=8
ifileM ='/home/mhaghigh/nc_file_in_outplume/so2_Iceland_2sep_0.7res.nc'
nc_so2 = Dataset(ifileM,'r')
so2 = nc_so2.variables['so2_TRL'][:,:]
lat_so2 = nc_so2.variables['lat'][:,:]
lon_so2 = nc_so2.variables['lon'][:,:]
lat_so2 = lat_so2[0:nlat,0:nlon]
lon_so2 = lon_so2[0:nlat,0:nlon]
so2 = so2[0:nlat,0:nlon]

so2_back = ma.masked_where(so2>1.0,so2)
mean_back = np.ma.mean(so2_back)
scale_fac = so2/1
test = ma.masked_where(so2 <=1.0,so2)
lon_m=ma.masked_where(so2 < 1.0, lon_so2)
scale_fac[scale_fac<1] = 1.0
scale_test = np.ma.filled(scale_fac,1)
print(np.shape(scale_test))

input_name_control = Dataset(file_control, 'r')
def read_mean(ccn, input_name):
    name = str(ccn)
    read_ccn = input_name.variables[name][8:16, :, :, :]*1e-6
    lat = input_name.variables['lat'][:]
    lon = input_name.variables['lon'][:]
    time_mean_ccn = np.ma.mean(read_ccn, axis=0)
    height_time_mean_ccn = np.ma.mean(time_mean_ccn, axis=0)
    print(np.shape(height_time_mean_ccn))
    return height_time_mean_ccn, lat, lon


def visualize(row, j, updraft,panels, avg_ccn):
    font_size = 40
    cmap = [(0.0, 0.0, 0.0)] + [(cm.jet(i)) for i in range(1, 256)]
    cmap = mpl.colors.ListedColormap(cmap)
    bounds = np.arange(0, 1100, 100)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap_2 = LinearSegmentedColormap.from_list("", ["lightskyblue", "steelblue", "green", "yellowgreen",
                                                    "yellow", "gold", "red", "firebrick", "darkred"])

    m = Basemap(ax=axs[row, j], projection='cyl', llcrnrlat=50, urcrnrlat=80, llcrnrlon=-60,
                urcrnrlon=20, resolution='c')
    m.drawcoastlines()
    m.drawmeridians(np.arange(-180., 180.,10.), linewidth=1.2, labels=[0, 0, 0, 1], fontsize = 20, color='black', zorder=3, latmax=90,
                 )
    m.drawparallels(np.arange(0., 85., 10.), linewidth=1.2, labels=[1, 0, 0, 0],  fontsize = 20, color='black', zorder=3, latmax=90,
                  )

    axs[row, j].set_title(updraft, fontsize=font_size,loc='center')
    axs[row, j].set_title(panels, loc='left', fontsize=font_size)

    axs[row, j] = m.imshow(avg_ccn, cmap=cmap_2, norm=norm)
    cax = fig.add_axes([0.15, 0.14, 0.69, 0.02]) #left, bottom, width,height
    cbar_bounds = bounds
    cbar_ticks = cbar_bounds
    cbar = fig.colorbar(axs[row, j], cax=cax, norm=norm, boundaries=cbar_bounds,
                        ticks=cbar_ticks, orientation='horizontal')
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label('CCN ($\mathrm{cm^{-3}}$)', fontsize=font_size)
    return


ccn1 = 'CCN_05'
ccn2 = 'CCN_07'
vertical_velocity_1 = '( w = 0.599 $\mathrm{m\,s^{-1}}$ )'
vertical_velocity_2 = '(w = 4.64 $\mathrm{m\,s^{-1}}$)'
control = ' no plume '
perturbed = ' plume '
panel_1 = control + vertical_velocity_1
panel_2 = perturbed + vertical_velocity_1
panel_3 = control + vertical_velocity_2
panel_4 = perturbed + vertical_velocity_2
panels_1 = '(a)'
panels_2 = '(b)'
panels_3 = '(c)'
panels_4 = '(d)'
fig, axs = plt.subplots(2, 2,figsize=(30,20))
#plt.tight_layout(rect=())
avg_ccn_con_2, lat, lon = read_mean(ccn1,input_name_control)
visualize(0, 0, panel_1, panels_1, avg_ccn_con_2)

avg_ccn_con_1, lat, lon = read_mean(ccn1, input_name_control)

avg_ccn_con_1 = avg_ccn_con_1*scale_test
visualize(0, 1, panel_2,panels_2, avg_ccn_con_1)



avg_ccn_per_1 , lat, lon = read_mean(ccn2,input_name_control)
visualize(1, 0, panel_3,panels_3,avg_ccn_per_1)

avg_ccn_per_2 , lat, lon = read_mean(ccn2,input_name_control)
avg_ccn_per_scaled_2 = avg_ccn_per_2*scale_test
visualize(1, 1, panel_4,panels_4,avg_ccn_per_scaled_2)
#plt.tight_layout(rect=(0.05,0.09,0.95,1))
plt.subplots_adjust(left=0.1,bottom=0.2,right=0.9, top=0.8, wspace=0.1, hspace=0.01)
#plt.subplot_tool()
plt.savefig('revised_paper_output/activated_ccn_fig3_new.pdf')
plt.savefig('revised_paper_output/activated_ccn_fig3_new.png')
plt.show()



