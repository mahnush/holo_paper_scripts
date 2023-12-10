import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
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
    so2_trl_in = so2_trl[so2_trl>1]
    so2_trm_in = so2_trm[so2_trm>1]
    so2_tru_in = so2_tru[so2_tru>1]
    so2_trl_mean = round(np.ma.mean(so2_trl_in, axis = 0),1)
    so2_trm_mean = round(np.ma.mean(so2_trm_in, axis = 0),1)
    so2_tru_mean = round(np.ma.mean(so2_tru_in, axis = 0),1)
    so2_trl_out = so2_trl[so2_trl<1]
    so2_trm_out = so2_trm[so2_trm<1]
    so2_tru_out = so2_tru[so2_tru<1]
    print(so2_tru_out)
    so2_trl_out_mean = np.ma.mean(so2_trl_out,  axis = 0)
    so2_trm_out_mean = np.ma.mean(so2_trm_out,  axis = 0)
    so2_tru_out_mean =np.ma.mean(so2_tru_out,  axis = 0)
    return so2_trl_mean,so2_trm_mean, so2_tru_mean, so2_trl_out_mean, so2_trm_out_mean, so2_tru_out_mean
so2_trl, so2_trm, so2_tru,so2_trl_out,so2_trm_out, so2_tru_out  =read_data(data_name)
#so2_mean_con = [so2_trl, so2_trm, 3]
print(so2_trl_out,so2_trm_out, so2_tru_out)
#height = np.arange(0,3)
#height2 = np.arange(3,9)
#height3 = np.arange(8,14)
#so2_test = np.zeros(3)
#so2_test2= np.zeros(6)
#so2_test3= np.zeros(6)+2.8
#so2_test[0:4] = so2_trl
#so2_test2[0:6] = so2_trm
#fs = 10
fig, ax = plt.subplots()
#ax.plot(so2_test, height, label = 'Lower Troposphere')
#ax.plot(so2_test2, height2, label = 'Middle Troposphere')
#ax.plot(so2_test3, height3, label = 'Upper Troposphere')
#ax.legend(fontsize = fs)
#y_tick = np.arange(1,16,2)
#x_thick = np.arange(2,6,1)
#ax.set_yticks(y_tick)
#ax.set_xticks(x_thick)
#ax.tick_params(axis='x',labelsize=fs)  # to Set Matplotlib Tick Labels Font Size
#ax.tick_params(axis='y', labelsize=fs)
#plt.xlabel("Mean value of vertical column SO2 inside of plume (DU)", fontsize = fs)
#plt.ylabel("Height (km)", fontsize = fs)
#ax.legend(fontsize = fs)
ax.bar(2.6, 3, width = 5.2, bottom=0, color = 'white', edgecolor = 'b')
ax.bar(1.9, 5, width = 3.8, bottom = 3, color = 'white', edgecolor = 'b' )
ax.bar(1, 5, width = 2, bottom = 8, color = 'white', edgecolor = 'b')
plt.show()
