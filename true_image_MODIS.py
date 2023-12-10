import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img_1sep = mpimg.imread('/home/mhaghigh/PycharmProjects/paper_Holo/true_image_MODIS/snapshot-2014-09-01T00_00_00Z.png')
img_2sep = mpimg.imread('/home/mhaghigh/PycharmProjects/paper_Holo/true_image_MODIS/snapshot-2014-09-02T00_00_00Z.png')
img_3sep = mpimg.imread('/home/mhaghigh/PycharmProjects/paper_Holo/true_image_MODIS/snapshot-2014-09-03T00_00_00Z.png')
img_4sep = mpimg.imread('/home/mhaghigh/PycharmProjects/paper_Holo/true_image_MODIS/snapshot-2014-09-04T00_00_00Z.png')
img_5sep = mpimg.imread('/home/mhaghigh/PycharmProjects/paper_Holo/true_image_MODIS/snapshot-2014-09-05T00_00_00Z.png')
img_6sep = mpimg.imread('/home/mhaghigh/PycharmProjects/paper_Holo/true_image_MODIS/snapshot-2014-09-06T00_00_00Z.png')
img_7sep = mpimg.imread('/home/mhaghigh/PycharmProjects/paper_Holo/true_image_MODIS/snapshot-2014-09-07T00_00_00Z.png')
fs = 40
print(np.shape(img_7sep))
fig, ax = plt.subplots(4, 2 ,figsize=(30, 20),sharex = True, sharey = True)
ax[0, 0].imshow(img_1sep)
ax[0, 0].set_title('1 September', fontsize=fs )
ax[0,0].set_yticklabels([])
ax[0,0].set_xticklabels([])
ax[0,1].imshow(img_2sep )
ax[0, 1].set_title('2 September', fontsize= fs)
ax[1,0].imshow(img_3sep)
ax[1, 0].set_title('3 September', fontsize = fs)
ax[1,1].imshow(img_4sep)
ax[1,1].set_title('4 September', fontsize = fs)
ax[2,0].imshow(img_5sep)
ax[2, 0].set_title('5 September', fontsize = fs)
ax[2,1].imshow(img_6sep)
ax[2, 1].set_title('6 September', fontsize = fs)
ax[3,0].imshow(img_7sep)
ax[3, 0].set_title('7 September', fontsize = fs)
fig.delaxes(ax[3,1])
plt.tight_layout()
#plt.subplots_adjust( hspace=0.05)
#plt.tight_layout()
#plt.subplots_adjust(wspace= 0.01)
#plt.tight_layout()
plt.savefig('MODIS_true_image.pdf', pad = 500)
plt.savefig('MODIS_true_image.png', pad = 500)
plt.show()
#def visulize(image_name, fig_title):
#    nrows = 4
#    ncols = 2
#    fig = plt.figure(figsize=(5*ncols, 5*nrows))
#    ax = plt.subplot(nrows, ncols)
#    ax.imshow(image_name, aspect = 'equal')
#    ax.set_title(fig_title)
#    return
#visulize(img_1sep, '1 September')
#plt.show()

