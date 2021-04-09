import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter,gaussian_filter
import scipy.sparse as sparse
import math
from scipy.ndimage import correlate
#from rescale_img import decompo_texture
from scipy.misc import imresize
from skimage.transform import resize
from skimage.util import dtype
import flow_operator as fo
import rescale_img as ri
##################################################################################
def detect_occlusion(Image1,Image2,u,v,h,b,sigma_d,sigma_i):
    div=np.gradient(u,axis=1)+np.gradient(v,axis=0)
    div[div>0]=0
    [Ix,Iy,It]=fo.derivatives(Image1,Image2,u,v,h,b)
    occ = np.exp(-div**2/2/sigma_d**2)*np.exp(-It**2/2/ sigma_i**2)
    return occ

####################################################################################
def denoise_color_weighted_medfilt2(u,v, im, occ, bfhsz, mfsz, sigma_i, fullVersion):
    sigma_x = 7   #  spatial distance (7)
    dilate_sz = (5, 5)  #dilation window size for flow edge region [5 5]
    sz = im.shape
    u0= median_filter(u, mfsz, mode='symmetric')
    v0=median_filter(v, mfsz, mode='symmetric')



