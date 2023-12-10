from scipy.stats import skew
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt


ipath = '/home/mhaghigh/nc_file_in_outplume/'
re_1sep = 'tqr_1sep.nc'
re_2sep = 'tqr_2sep.nc'
re_3sep = 'tqr_3sep.nc'
re_4sep = 'tqr_4sep.nc'
re_5sep = 'tqr_5sep.nc'
#re_6sep = 'alb_6sep_1d.nc'
#re_7sep = 'alb_7sep_1d.nc'
ipath_m = '/home/mhaghigh/nc_file_in_outplume/'
re_1sep_m = 'modis_lwp_1sep.nc'
re_2sep_m = 'modis_lwp_2sep.nc'
re_3sep_m = 'modis_lwp_3sep.nc'
re_4sep_m = 'modis_lwp_4sep.nc'
re_5sep_m = 'modis_lwp_5sep.nc'
mdata1_m = ipath_m + re_1sep_m
mdata2_m = ipath_m + re_2sep_m
mdata3_m = ipath_m + re_3sep_m
mdata4_m = ipath_m + re_4sep_m
mdata5_m = ipath_m + re_5sep_m

mdata1 = ipath + re_1sep
mdata2 = ipath + re_2sep
mdata3 = ipath + re_3sep
mdata4 = ipath + re_4sep
mdata5 = ipath + re_5sep
#mdata6 = ipath + re_6sep
#mdata7 = ipath + re_7sep
data_name = [mdata1, mdata2, mdata3, mdata4, mdata5]
data_name_modis = [mdata1_m, mdata2_m, mdata3_m, mdata4_m, mdata5_m]


def read_data(mdata1):
    nc_m_1 = Dataset(mdata1, 'r')
    nc_m_1_m = Dataset(mdata1_m, 'r')
    re_1_in = nc_m_1.variables['re_per_in'][:]
    re_1_out = nc_m_1.variables['re_per_out'][:]
    re_1_c_in = nc_m_1.variables['re_con_in'][:]
    re_1_c_out = nc_m_1.variables['re_con_out'][:]
    return re_1_in, re_1_out, re_1_c_in, re_1_c_out
def read_data_modis(mdata1_m):
    nc_m_1_m = Dataset(mdata1_m, 'r')
    re_1_in_m = nc_m_1_m.variables['re_per_in'][:]
    re_1_out_m = nc_m_1_m.variables['re_per_out'][:]
    #re_1_out_m_n =np.ma.masked_where(re_1_out_m>1002,re_1_out_m)
    re_1_out_m_n = re_1_out_m.compressed()
    #re_1_in_m_n = np.ma.masked_where(re_1_in_m>1002,re_1_in_m)
    re_1_in_m_n = re_1_in_m.compressed()
    return re_1_in_m_n, re_1_out_m_n


allre_in = []
allre_out = []
allre_in_c = []
allre_out_c = []
allre_in_m = []
allre_out_m = []

for data in data_name:
    re_1_in, re_1_out, re_1_c_in, re_1_c_out = read_data(data)
    allre_in = np.concatenate((allre_in, re_1_in), axis=0)
    allre_out = np.concatenate((allre_out, re_1_out), axis=0)
    allre_in_c = np.concatenate((allre_in_c, re_1_c_in), axis=0)
    allre_out_c = np.concatenate((allre_out_c, re_1_c_out), axis=0)

allre_in = allre_in*1000
allre_out = allre_out*1000
allre_in_c = allre_in_c*1000
allre_out_c = allre_out_c*1000
for data in data_name_modis:
    re_1_in_m, re_1_out_m = read_data_modis(data)
    allre_in_m = np.concatenate((allre_in_m, re_1_in_m), axis=0)
    allre_out_m = np.concatenate((allre_out_m, re_1_out_m), axis=0)


print(np.amax(allre_in_m))
print(np.amax(allre_out_m))


def weight(var):
    #weight = (1 + np.zeros(len(var))) / len(var)
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight



def lable_hist(var):
    median = str(np.median(var))
    mean = str(np.ma.mean(var))
    std = str(np.std(var))
    skew_l = str(skew(var))
    #print(np.median(var))
    #print(np.mean(var))
    #print(std)
    #print(skew_l)
    lable = '('+'mean = ' + median + ')'

    return lable
fig, (axs0, axs1) = plt.subplots(1, 2, figsize=(30, 20), sharex=True, sharey=True)
font_legend = 40
font_lable = 40
numbin = np.arange(1,800,10)
line_width = 4
font_tick = 40
name = '$ \mathrm{RWP}$ ($\mathrm{g\,m^{-2}}$)'
axs0.hist(allre_in,bins=numbin, density=True , histtype='step',
          linewidth=line_width,color = 'red', log = True, label='Volcano ')#+lable_hist(allre_in))

axs0.hist(allre_in_c,  bins=numbin, density=True,  histtype='step',
          linewidth=line_width, color = 'blue', log = True, label='No-Volcano ')#+lable_hist(allre_in_c))
#axs0.hist(allre_in_m, bins=numbin, density=True,  histtype='step',
#          linewidth=line_width, color = 'black',  label='Modis '+ lable_hist(allre_in_m))
axs0.legend(loc='upper right', fontsize=font_legend, frameon=True)
ticks = np.arange(0, 0.012, 0.002)
ticks_x = np.arange(0,1000,100)
#axs0.set_yticks(ticks)
axs0.tick_params(axis='x', labelsize=font_tick)  # to Set Matplotlib Tick Labels Font Size
axs0.tick_params(axis='y', labelsize=font_tick)
axs0.set_xlabel(name, fontsize=font_lable)
axs0.set_ylabel('Relative Frequency', fontsize=font_lable)
axs1.hist(allre_out, bins=numbin, density=True,  histtype='step',
          linewidth=line_width, color = 'red',log = True, label='Volcano ')#+lable_hist(allre_out))
axs1.hist(allre_out_c, bins=numbin, density=True,  histtype='step',
          linewidth=line_width, color = 'blue', log = True, label='No-Volcano ')#+lable_hist(allre_out_c))
#axs1.hist(allre_out_m, bins=numbin, density=True, histtype='step',
#          linewidth=line_width, color = 'black',label='Modis ' + lable_hist(allre_out_m))
axs1.legend(loc='upper right', fontsize=font_legend, frameon=True)
#axs1.set_yticks(ticks)
#axs0.set_xticks(ticks_x)
#axs1.set_xticks(ticks_x)
axs1.tick_params(axis='x', labelsize=font_tick)  # to Set Matplotlib Tick Labels Font Size
#axs1.tick_params(axis='y', labelsize=font_tick)
#axs1.yticks( " ")
#axs1.set_yticklabels([])
axs1.set_xlabel(name, fontsize=font_lable)
#axs1.set_ylabel('probability density function', fontsize=font_lable)

axs0.set_title('Inside Plume', fontsize= font_lable)
axs1.set_title('Outside Plume', fontsize=font_lable)
axs0.annotate('(a)',xy=(-20,0.035),size=font_lable)
axs1.annotate('(b)',xy=(-30,0.05),size=font_lable)
plt.tight_layout()
axs0.grid(True)
axs1.grid(True)
#plt.savefig('inside_outside_outputs/tqr_no_label_fig6.png')
plt.savefig('/home/mhaghigh/PycharmProjects/paper_Holo/revised_paper_output/tqr_no_label_fig6.pdf')
plt.savefig('/home/mhaghigh/PycharmProjects/paper_Holo/revised_paper_output/tqr_no_label_fig6.png')

plt.show()
