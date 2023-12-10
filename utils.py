def grid_cordinate(minlat, minlon, maxlat, maxlon):
    xdim = int(1 + (maxlon - minlon) / dx)
    ydim = int(1 + ((maxlat - minlat) / dy))
    grid_lon = np.zeros((xdim, ydim))
    grid_lat = np.zeros((xdim, ydim))
    for i in range(xdim):
        for j in range(ydim):
            grid_lon[i, j] = dx*i+minlon
            grid_lat[i, j] = dx*j+minlat
    return grid_lon, grid_lat

def grid(minlat, minlon, maxlat, maxlon, in_var, in_lon, in_lat):
    xdim = int(1 + (maxlon - minlon) / dx)
    ydim = int(1 + ((maxlat - minlat) / dy))
    print(xdim, ydim)
    sum_var = np.zeros((xdim, ydim))
    count = np.zeros((xdim, ydim))
    in_var_mask = np.ma.filled(in_var,0)
    for ii in range(len(in_var)):
        if (in_lat[ii] >= minlat and in_lat[ii] <= maxlat and in_lon[ii] >= minlon and in_lon[ii] <= maxlon):
            i = round((in_lon[ii] - minlon) / dx)
            i = int(i)
            j = round((in_lat[ii] - minlat) / dy)
            j = int(j)
            sum_var[i, j] = sum_var[i, j] + in_var_mask[ii]
            if in_var[ii] == 0:
                count[i, j] = count[i, j]+0
            else:
                count[i, j] = count[i, j]+1
    count = np.ma.masked_equal(count, 0)
    avg_var = sum_var/count

    return avg_var


grid_lon,grid_lat = grid_cordinate(minlat, minlon, maxlat, maxlon)
incoming_sw_flux = grid(minlat, minlon, maxlat, maxlon, sw_flux_inward, lon, lat)
upward_sw_flux = grid(minlat, minlon, maxlat, maxlon, sw_flux_upward, lon, lat)
albedo = upward_sw_flux/incoming_sw_flux
print(np.amax(albedo))
cmap = [(0.0,0.0,0.0)] + [(cm.jet(i)) for i in range(1,256)]
cmap = mpl.colors.ListedColormap(cmap)
bounds = np.arange(0,1.1,0.1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cmap_2 = LinearSegmentedColormap.from_list("", ["white","lightskyblue","steelblue","green","yellowgreen","yellow","gold","red","firebrick","darkred"])
fig, axs0 = plt.subplots(1,1,figsize=(30,20))
m = Basemap(projection='cyl',llcrnrlat=50,urcrnrlat=80,\
           llcrnrlon=-60,urcrnrlon=20,resolution='c')

m.drawcoastlines()
x, y = m(grid_lon,grid_lat)
axs0.set_title('albedo')
axs0=m.pcolormesh(x, y, albedo,cmap=cmap_2, norm=norm)

cax = fig.add_axes([0.15,0.1,0.7,0.02])
cbar_bounds = bounds
cbar_ticks =  bounds
cbar = fig.colorbar(axs0,cax=cax, norm=norm, boundaries=cbar_bounds, ticks=cbar_ticks, orientation='horizontal')
cbar.set_label('albedo', fontsize=30)
cbar.ax.tick_params(labelsize='x-large')
plt.show()
