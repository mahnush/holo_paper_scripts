import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
file_20km = '/home/mhaghigh/manuscript/radiation_model/NWP_LAM_DOM01_20140902T100000Z_0034_20km.nc'
file_2km = '/home/mhaghigh/manuscript/radiation_model/NWP_LAM_DOM01_20140902T100000Z_0034.nc'

#input_name_20km = Dataset(file_20km, 'r')
#input_name_2km = Dataset(file_2km, 'r')

def read_nc(file_name, var_name):
    from netCDF4 import Dataset
    file = file_name
    nc = Dataset(file,'r')
    var = nc.variables[var_name][:]
    print(np.shape(var))
    return var[0,:,:]

def visulize_model(ax, var, bounds, cbar_label, titel_figure, map_limit):
    fs_titel = 20
    fs_label = 20
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    cmap = [(0.0,0.0,0.0)] + [(cm.jet(i)) for i in range(1,256)]
    cmap = mpl.colors.ListedColormap(cmap)
    bounds = bounds
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    #cmap_2 = LinearSegmentedColormap.from_list("", ["white","lightskyblue","steelblue","green","yellowgreen","yellow","gold","red","firebrick","darkred"])
    cmap_2 = LinearSegmentedColormap.from_list("", [ "lightskyblue", "steelblue", "green", "yellowgreen", "yellow",
                                                "gold", "red", "firebrick", "darkred"])
    m = Basemap(ax=ax, projection='cyl', llcrnrlat= map_limit[0], urcrnrlat=map_limit[1],\
           llcrnrlon=map_limit[2], urcrnrlon=map_limit[3], resolution='c')
    m.drawmeridians(np.arange(-180., 180.,5.), linewidth=1.2, labels=[0, 0, 0, 1], color='grey', zorder=3, latmax=90,
                 )
    m.drawparallels(np.arange(0., 85., 5.), linewidth=1.2, labels=[1, 0, 0, 0], color='grey', zorder=2, latmax=90,
                  )
    m.drawcoastlines()
    ax.set_title(titel_figure, fontsize=fs_titel)
    l1 =m.imshow(var, cmap = cmap_2, norm = norm)
    cbar = plt.colorbar(l1, ax=ax)
    cbar_bounds = bounds
    cbar_ticks =  bounds
    cbar.set_label(cbar_label, fontsize= fs_label)
    cbar.ax.tick_params(labelsize='xx-large')

rad_20km = read_nc(file_20km, 'sob_t')
rad_2km = read_nc(file_2km, 'sob_t')

fig, ((ax0,ax1)) = plt.subplots(2, 1, figsize=(17, 12))
bounds = np.arange(0,1000,100)
cbar_lable = 'Net Short Flux at TOA (w/m2)'
titel_figure_2km = '2.5km Resolution'
titel_figure_20km = '20km Resolution'
map_limit = [50,80,-60,20]
visulize_model(ax0, rad_2km, bounds, cbar_lable, titel_figure_2km,map_limit)
visulize_model(ax1, rad_20km, bounds, cbar_lable, titel_figure_20km, map_limit)
plt.savefig('radiation_plots.pdf')
plt.savefig('radiation_plots.png')

plt.show()