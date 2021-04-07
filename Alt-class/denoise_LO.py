import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import scipy.sparse as sparse
import math
from scipy.ndimage import correlate
#####################################################
def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    #If the number of lines # Let A m × n, for the [p q] of the block division, the final matrix of p × q, is the number of columns (m-p + 1) × (n-q + 1).
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    # Moved along the line, so the first holding column (i) does not move down along the row (j)
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
    return result
######################################################
def denoise_LO (un, median_filter_size, lambda23, niters):
    mfsize = median_filter_size
    hfsize = math.floor(mfsize/2)
    n   = (mfsize*mfsize-1)/2
  
    tmp = np.arange(-n,n+1,dtype=np.float32)
    tmp = np.matlib.repmat( tmp, un.shape[0]*un.shape[1],1)/lambda23
    tmp=tmp.T
    tmp = np.matlib.repmat( np.reshape(un,(1,un.shape[0]*un.shape[1]),'F') ,int(2*n+1), 1)+tmp
    print(tmp)

    uo  = un
    for i in range(niters):
        u=np.pad(uo,((hfsize,hfsize),(hfsize,hfsize)),mode='symmetric')
        u2 = im2col(u, (mfsize,mfsize))
        #print(u)


    return uo

#un=np.random.rand(3,4)
un=np.array([[1 ,2 ,3 ,4 ], [5 ,6 ,7 ,8 ] ,[9 ,10 ,11 ,12 ],[13, 14,15,16]])
median_filter_size=2
lambda23=1
niters=1
u0=denoise_LO (un, median_filter_size, lambda23, niters)
print(u0)