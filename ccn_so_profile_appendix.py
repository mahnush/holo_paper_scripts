import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
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
print(mean_back)
scale_fac = so2/0.5
test = ma.masked_where(so2 <=1.0,so2)
lon_m=ma.masked_where(so2 < 1.0, lon_so2)
scale_fac[scale_fac<1] = 1.0
scale_test = np.ma.filled(scale_fac,1)

input_name_control = Dataset(file_control, 'r')
def read_mean(ccn, input_name):
    name = str(ccn)
    read_ccn = input_name.variables[name][8:16, :, :, :]*1e-6
    lat = input_name.variables['lat'][:]
    lon = input_name.variables['lon'][:]
    height = input_name.variables['z'][8:16, :, :, :]
    time_mean_ccn = np.ma.mean(read_ccn, axis=0)
    return time_mean_ccn, lat, lon, height
ccn1 = 'CCN_05'
ccn2 = 'CCN_10'
vertical_velocity_1 = '( w = 0.599 $\mathrm{m\,s^{-1}}$ )'
vertical_velocity_2 = '(w = 4.64 $\mathrm{m\,s^{-1}}$)'
control = ' no plume '
perturbed = ' plume '
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
fig, axs = plt.subplots(2, 2,figsize=(10,8))

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
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

avg_ccn_con_2, lat, lon, height = read_mean(ccn1,input_name_control)
print(np.shape(height))
height_temp_mean = np.mean(height,axis =0)
height_lat_mean = np.mean(height_temp_mean,axis = 1)
height_mean = np.mean(height_lat_mean,axis = 1)
print('height')
height_axis = np.round(height_mean)*1e-3
avg_ccn_per_2 = avg_ccn_con_2 * scale_test
avg_ccn_con_inside = np.zeros((60))
avg_ccn_per_inside = np.zeros((60))
avg_ccn_outside = np.zeros((60))
for ik in range(60):
  temp_con_inside = avg_ccn_con_2[ik,:,:][scale_test>1]
  avg_ccn_con_inside[ik] = np.mean(temp_con_inside, axis = 0)
  temp_con_outside = avg_ccn_con_2[ik,:,:][scale_test<=1]
  avg_ccn_outside[ik] = np.mean(temp_con_outside, axis = 0)
  temp_per_inside = avg_ccn_per_2[ik,:,:][scale_test>1]
  avg_ccn_per_inside[ik] = np.mean(temp_per_inside, axis = 0)


lev = np.arange(1,61)
print(height_axis[30])
avg_ccn_con_inside = avg_ccn_con_inside[30:59]
avg_ccn_per_inside = avg_ccn_per_inside[30:59]
avg_ccn_outside = avg_ccn_outside[30:59]
height_axis = height_axis[30:59]
fs_axis = 50
fs = 40
fig, (ax1, ax) = plt.subplots(1, 2, figsize=(30, 20))
ax.plot(avg_ccn_con_inside, height_axis, label="no-volcano (in-plume)",linewidth=4)
ax.plot(avg_ccn_per_inside, height_axis, label="volcano (in-plume)", linewidth=4)
ax.plot(avg_ccn_outside, height_axis, label="no-volcano (out-plume)", linewidth=4)
y_tick = np.arange(0,12,2)
x_thick = np.arange(0,5000,1000)
ax.set_yticks(y_tick)
ax.set_xticks(x_thick)
ax.tick_params(axis='x',labelsize=fs)  # to Set Matplotlib Tick Labels Font Size
ax.tick_params(axis='y', labelsize=fs)
ax.grid()
ax.set_xlabel("CCN ($\mathrm{cm^{-3}}$)", fontsize = fs)
ax.set_ylabel("Height (km)", fontsize = fs)
ax.legend(fontsize = fs)

data_name = 'so2_Iceland_2sep_0.7res.nc'

def read_data(data_name):
    input_path = '/home/mhaghigh/nc_file_in_outplume/'
    input_name = input_path + data_name
    input_cams = Dataset(input_name, 'r')
    so2_trl = input_cams.variables['so2_TRL'][:, :]
    so2_trm = input_cams.variables['so2_TRM'][:, :]
    so2_tru = input_cams.variables['so2_TRU'][:, :]
    lat = input_cams.variables['lat'][:,:]
    lon = input_cams.variables['lon'][:,:]
    #so2 = np.ma.masked_where(sulfat_con<1,sulfat_con)
    so2_trl = so2_trl[so2_trl>1]
    so2_trm = so2_trm[so2_trm>1]
    so2_tru = so2_tru[so2_tru>1]
    so2_trl_mean = round(np.ma.mean(so2_trl, axis = 0),1)
    so2_trm_mean = round(np.ma.mean(so2_trm, axis = 0),1)
    so2_tru_mean = round(np.ma.mean(so2_tru, axis = 0),1)
    return so2_trl_mean,so2_trm_mean, so2_tru_mean
so2_trl, so2_trm, so2_tru =read_data(data_name)
so2_mean_con = [so2_trl, so2_trm, 3]

height = np.arange(0,3)
height2 = np.arange(3,9)
height3 = np.arange(8,14)
so2_test = np.zeros(3)
so2_test2= np.zeros(6)
so2_test3= np.zeros(6)+2.8
so2_test[0:4] = so2_trl
so2_test2[0:6] = so2_trm


# ax1.plot(so2_test, height, label = 'Lower Troposphere', linewidth = 4,  color =lighten_color('b', 1.6) )
# ax1.plot(so2_test2, height2, label = 'Middle Troposphere', linewidth = 4, color = 'b')
# ax1.plot(so2_test3, height3, label = 'Upper Troposphere', linewidth = 4, color =lighten_color('b', 0.4))
#ax1.bar(2.6, 3, width = 5.2, bottom=0, color = 'white', edgecolor = 'b')
ax1.bar(2.6, 3, width = 5.2, bottom=0, edgecolor = lighten_color('b', 1.6), color= 'white',linewidth = 4,  label = 'Lower Troposphere' )
ax1.bar(1.9, 5, width = 3.8, bottom = 3, edgecolor = 'b' , color= 'white',linewidth = 4,label = 'Middle Troposphere' )
ax1.bar(1, 5, width = 2, bottom = 8, edgecolor = lighten_color('b', 0.4),  color= 'white',linewidth = 4, label = 'Upper Troposphere' )

ax1.legend(fontsize = fs)
y_tick = np.arange(1,16,2)
x_thick = np.arange(0,7,1)
ax1.set_yticks(y_tick)
ax1.set_xticks(x_thick)
ax1.tick_params(axis='x',labelsize=fs)  # to Set Matplotlib Tick Labels Font Size
ax1.tick_params(axis='y', labelsize=fs)
ax1.set_xlabel("Mean value of column SO2 in-plume (DU)", fontsize = fs)
ax1.set_ylabel("Height (km)", fontsize = fs)
ax1.legend(fontsize = fs)
ax.annotate('(b)',xy=(-15,11),size=fs)
ax1.annotate('(a)',xy=(-0.04,14.4),size=fs)

plt.savefig('Appendix_CCN_profile.pdf')
plt.savefig('Appendix_CCN_profile.png')
plt.show()