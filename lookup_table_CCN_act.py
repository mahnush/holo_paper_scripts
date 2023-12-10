import numpy as np
from netCDF4 import Dataset
file_control = '/home/mhaghigh/1st_paper/ncfile/holo_test_fig3.nc'

input_name_control = Dataset(file_control, 'r')
def read_mean(ccn_name, input_name):
    read_ccn = input_name.variables[ccn_name][16:24, 48, :, :]*1e-6
    lat = input_name.variables['lat'][:]
    lon = input_name.variables['lon'][:]
    height = input_name.variables['z'][8:16, 49 ,:, :]
    height = height.flatten()
    #print(np.mean(height))
    time_mean_ccn = np.ma.mean(read_ccn, axis=0)
    time_mean_ccn = time_mean_ccn.flatten()
    Q1 = np.quantile(time_mean_ccn, 0.25)
    Q3 = np.quantile(time_mean_ccn, 0.75)
    mean = np.mean(time_mean_ccn)
    return print('Q1 =', round(Q1), ',Q3 = ', round(Q3), ',mean = ',  round(mean))
print('CCN01')
read_mean('CCN_01' , input_name_control)
print('CCN02')
read_mean('CCN_02' , input_name_control)
print('CCN03')
read_mean('CCN_03' , input_name_control)
print('CCN04')
read_mean('CCN_04' , input_name_control)
print('CCN05')
read_mean('CCN_05' , input_name_control)
print('CCN06')
read_mean('CCN_06' , input_name_control)
print('CCN07')
read_mean('CCN_07' , input_name_control)
print('CCN08')
read_mean('CCN_08' , input_name_control)
print('CCN09')
read_mean('CCN_09' , input_name_control)
print('CCN10')
read_mean('CCN_10' , input_name_control)