def read_RGB(file_name):
    from pyhdf.SD import SD, SDC
    import numpy as np
    from netCDF4 import Dataset
    file = SD(file_name, SDC.READ)
    #print(list(file.keys()))
    #print(list(file['']))
    selected_sds = file.select('EV_250_Aggr1km_RefSB')

    selected_sds_attributes = selected_sds.attributes()

    for key, value in selected_sds_attributes.items():
        #print key, value
        if key == 'reflectance_scales':
            reflectance_scales_250_Aggr1km_RefSB = np.asarray(value)
        if key == 'reflectance_offsets':
            reflectance_offsets_250_Aggr1km_RefSB = np.asarray(value)

    sds_data_250_Aggr1km_RefSB = selected_sds.get()

    print('sds_data shape', sds_data_250_Aggr1km_RefSB.shape)
    print(reflectance_scales_250_Aggr1km_RefSB.shape)


    selected_sds = file.select('EV_500_Aggr1km_RefSB')

    selected_sds_attributes = selected_sds.attributes()

    for key, value in selected_sds_attributes.items():
        if key == 'reflectance_scales':
            reflectance_scales_500_Aggr1km_RefSB = np.asarray(value)
        if key == 'reflectance_offsets':
            reflectance_offsets_500_Aggr1km_RefSB = np.asarray(value)

    sds_data_500_Aggr1km_RefSB = selected_sds.get()

    print( reflectance_scales_500_Aggr1km_RefSB.shape)


    data_shape = sds_data_250_Aggr1km_RefSB.shape

    along_track = data_shape[1]
    cross_trak = data_shape[2]

    z = np.zeros((along_track, cross_trak,3))

    z[:,:,0] = ( sds_data_250_Aggr1km_RefSB[0,:,:] - reflectance_offsets_250_Aggr1km_RefSB[0] ) * reflectance_scales_250_Aggr1km_RefSB[0]
    z[:,:,1] = ( sds_data_500_Aggr1km_RefSB[1,:,:] - reflectance_offsets_500_Aggr1km_RefSB[1] ) * reflectance_scales_500_Aggr1km_RefSB[1]
    z[:,:,2] = ( sds_data_500_Aggr1km_RefSB[0,:,:] - reflectance_offsets_500_Aggr1km_RefSB[0] ) * reflectance_scales_500_Aggr1km_RefSB[0]


    norme = 0.4 # factor to increase the brightness ]0,1]

    rgb = np.zeros((along_track, cross_trak,3))

    rgb = z / norme

    rgb[ rgb > 1 ] = 1.0
    rgb[ rgb < 0 ] = 0.0
    rgb = np.fliplr(rgb)
    print(np.shape(rgb))
    rgb1 = rgb[0,:,:].flatten()
    rgb2 = rgb[1,:,:].flatten()
    rgb3 = rgb[2,:,:].flatten()
    return rgb1, rgb2, rgb3
def grid_coordinate(limit,gridSize ):
    minlat = float(limit[0])
    maxlat = float(limit[1])
    minlon = float(limit[2])
    maxlon = float(limit[3])
    dx = gridSize
    xdim=int(1+((maxlon-minlon)/dx))
    ydim=int(1+((maxlat-minlat)/dx))
    grdlat=np.full([xdim,ydim],-1.0)
    grdlon=np.full([xdim,ydim],-1.0)
    for i in range(xdim):
        for j in range(ydim):
            grdlon[i,j]=dx*i+minlon
            grdlat[i,j]=dx*j+minlat
    return grdlat,grdlon
def read_coordinate(file_gn):
    from pyhdf.SD import SD, SDC
    file_g = SD(file_gn, SDC.READ)
    lat_2D = file_g.select('Latitude')
    lon_2D=file_g.select('Longitude')
    lat=lat_2D.get()
    lon=lon_2D.get()
    latitude=lat.flatten()
    longitude=lon.flatten()
    return latitude, longitude
def grid(limit, gsize, indata, inlat, inlon):
    dx = gsize
    dy = gsize
    minlat = float(limit[0])
    maxlat = float(limit[1])
    minlon = float(limit[2])
    maxlon = float(limit[3])
    xdim = int(1 + ((maxlon - minlon) / dx))
    ydim = int(1 + ((maxlat - minlat) / dy))
    sum_var  = np.zeros((xdim, ydim))
    count = np.full([xdim, ydim],1)
    avg_var = np.full([xdim, ydim], -1.0)

    mask_re = np.where(indata != 0, 1, 0)

    for ii in range(len(indata)):

        if (inlat[ii] >= minlat and inlat[ii] <= maxlat and inlon[ii] >= minlon and inlon[ii] <= maxlon):

            i = round((inlon[ii] - minlon) / dx)
            i = int(i)
            j = round((inlat[ii] - minlat) / dy)
            j = int(j)
            sum_var[i, j] = sum_var[i, j] + indata[ii]
            #count[i, j] += mask_re[ii]
            count[i,j] = count[i,j] + 1

    #count = np.ma.masked_equal(count, 0)
    avg_var = sum_var / count

    #avg_var = np.ma.masked_equal(avg_var, -1)

    return (avg_var)
import numpy as np
# --set the limit for Lat and Lon and grid size
limit = np.zeros(4)
limit[0] = 50
limit[1] = 80
limit[2] = -60
limit[3] = 20
gridSize = 0.01

# --opening L2 and geo file list
fileList = open('/home/mhaghigh/RGB_HV_data/file_list.txt', 'r+')
fileList_geo = open('/home/mhaghigh/RGB_HV_data/g_file_list.txt', 'r+')
geo_file = [line for line in fileList_geo.readlines()]
latgrid, longrid = grid_coordinate(limit, gridSize)
# --defining the array to put all lat and lon and data
allLat = []
allLon = []
allrgb1 = []
allrgb2 = []
allrgb3 = []
ipath = '/home/mhaghigh/RGB_HV_data/'
k = 0

for FILE_NAME in fileList:
    FILE_NAME = FILE_NAME.strip()
    file_gn = geo_file[k]
    file_gn = file_gn.strip()
    # print(k)
    FILE_NAME = ipath + FILE_NAME
    print(FILE_NAME)
    file_gn = ipath + file_gn
    print(file_gn)
    if len(read_RGB(FILE_NAME)[0]) == 0:
        allLat, allLon = read_coordinate(file_gn)
        allrgb1 = read_RGB(FILE_NAME)[0]
        allrgb2 = read_RGB(FILE_NAME)[1]
        allrgb3 = read_RGB(FILE_NAME)[3]
    elif len(read_RGB(FILE_NAME)[0]) > 0:
        allLat = np.concatenate((allLat, read_coordinate(file_gn)[0]), axis=0)
        allLon = np.concatenate((allLon, read_coordinate(file_gn)[1]), axis=0)
        allrgb1 = np.concatenate((allrgb1, read_RGB(FILE_NAME)[0]), axis=0)
        allrgb2 = np.concatenate((allrgb2, read_RGB(FILE_NAME)[1]), axis=0)
        allrgb3 = np.concatenate((allrgb3, read_RGB(FILE_NAME)[2]), axis=0)
print(np.amax(allrgb2))
print(np.amin(allrgb2))
print(allrgb2)
rgb_grid1 = grid(limit, gridSize, allrgb1, allLat, allLon)
rgb_grid2 = grid(limit, gridSize, allrgb2, allLat, allLon)
rgb_grid3 = grid(limit, gridSize, allrgb3, allLat, allLon)
print(np.shape(rgb_grid1))
print(np.shape(rgb_grid2))
print(np.shape(rgb_grid3))
nlon = 8001
nlat = 3001
rgb_grid_all = np.zeros((nlon, nlat,3))
rgb_grid_all[:,:,0] = rgb_grid1
rgb_grid_all[:,:,1] = rgb_grid2
rgb_grid_all[:,:,2] = rgb_grid3
print(np.nanmax(rgb_grid2))
import matplotlib.pyplot as plt
fig = plt.figure()

ax = fig.add_subplot(111)

img = plt.imshow(np.fliplr(rgb_grid_all),  origin='lower')
plt.show()