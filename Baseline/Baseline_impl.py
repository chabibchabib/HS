import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2 as cv
import math
from scipy.ndimage.filters import convolve as filter2
import warping_fct as mod


######################################################################

######################################################################
beforeImg = cv.imread('img_000000_18.02244.tiff', 0).astype(float)
afterImg = cv.imread('img_000050_30.42245.tiff', 0).astype(float)

print('image0 shape',afterImg.shape)
l=mod.determine_numLevels(afterImg,1/0.8)
#l=3
print(l)
h=np.array([[-1, 8, 0, -8, 1]],np.float32)
h=1/12*h
u,v = mod.warping_pyram(beforeImg, afterImg,factor=1/0.8, alpha = 15, delta = 10**-1,Level=l,kernel_size=5,downsampling_gauss=1/0.8,h=h,b=0.1)
mod.draw_quiver(u, v, beforeImg)



