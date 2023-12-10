import numpy as np

def grid(limit, gsize, indata, inlat, inlon):
    dx = gsize
    dy = gsize
    minlat = float(limit[0])
    maxlat = float(limit[1])
    minlon = float(limit[2])
    maxlon = float(limit[3])
    xdim = int(1 + ((maxlon - minlon) / dx))
    ydim = int(1 + ((maxlat - minlat) / dy))
    print(xdim,ydim)
    sum_var  = np.zeros((xdim, ydim))
    count = np.zeros((xdim, ydim))
    avg_var = np.full([xdim, ydim], -1.0)

    indata[indata < 0] = 0
    indata[indata>1e38] = 0
    mask_re = np.where(indata != 0, 1, 0)
    for ii in range(len(indata)):

        if (inlat[ii] >= minlat and inlat[ii] <= maxlat and inlon[ii] >= minlon and inlon[ii] <= maxlon):
            i = round((inlon[ii] - minlon) / dx)
            i = int(i)
            j = round((inlat[ii] - minlat) / dy)
            j = int(j)
            sum_var[i, j] = sum_var[i, j] + indata[ii]
            count[i, j] += mask_re[ii]
            #count[i, j] += 1
    count = np.ma.masked_equal(count, 0)
    avg_var = sum_var / count

    avg_var = np.ma.masked_equal(avg_var, -1)

    return (avg_var)
