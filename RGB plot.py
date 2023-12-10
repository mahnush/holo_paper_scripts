from pyhdf.SD import SD, SDC

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pprint

#----------------------------------------------------------------------------------------#
# inputs

file_name = '/home/mhaghigh/RGB_HV_data/MYD021KM.A2014244.1645.061.2018055015608.hdf'

file = SD(file_name, SDC.READ)


#----------------------------------------------------------------------------------------#

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

#----------------------------------------------------------------------------------------#

selected_sds = file.select('EV_500_Aggr1km_RefSB')

selected_sds_attributes = selected_sds.attributes()

for key, value in selected_sds_attributes.items():
	if key == 'reflectance_scales':
		reflectance_scales_500_Aggr1km_RefSB = np.asarray(value)
	if key == 'reflectance_offsets':
		reflectance_offsets_500_Aggr1km_RefSB = np.asarray(value)

sds_data_500_Aggr1km_RefSB = selected_sds.get()

print( reflectance_scales_500_Aggr1km_RefSB.shape)

#----------------------------------------------------------------------------------------#

data_shape = sds_data_250_Aggr1km_RefSB.shape

along_track = data_shape[1]
cross_trak = data_shape[2]

z = np.zeros((along_track, cross_trak,3))

z[:,:,0] = ( sds_data_250_Aggr1km_RefSB[0,:,:] - reflectance_offsets_250_Aggr1km_RefSB[0] ) * reflectance_scales_250_Aggr1km_RefSB[0]
z[:,:,1] = ( sds_data_500_Aggr1km_RefSB[1,:,:] - reflectance_offsets_500_Aggr1km_RefSB[1] ) * reflectance_scales_500_Aggr1km_RefSB[1]
z[:,:,2] = ( sds_data_500_Aggr1km_RefSB[0,:,:] - reflectance_offsets_500_Aggr1km_RefSB[0] ) * reflectance_scales_500_Aggr1km_RefSB[0]

#----------------------------------------------------------------------------------------#

norme = 0.4 # factor to increase the brightness ]0,1]

rgb = np.zeros((along_track, cross_trak,3))

rgb = z / norme

rgb[ rgb > 1 ] = 1.0
rgb[ rgb < 0 ] = 0.0
print(np.shape(rgb))
#----------------------------------------------------------------------------------------#
# plot image using matplotlib

fig = plt.figure()

ax = fig.add_subplot(111)

img = plt.imshow(np.fliplr(rgb), interpolation='nearest', origin='lower')

l = [int(i) for i in np.linspace(0,cross_trak,6)]
plt.xticks(l, [i for i in reversed(l)], rotation=0, fontsize=7 )

l = [int(i) for i in np.linspace(0,along_track,9)]
plt.yticks(l, l, rotation=0, fontsize=7 )

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.title('How to plot a MODIS RGB image \n using python 3 ?', fontsize=8)

#plt.savefig("modis_granule_rgb2.png", bbox_inches='tight', dpi=100)

plt.show()

plt.close()