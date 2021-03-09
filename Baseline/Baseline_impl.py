import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2 as cv
import math
from scipy.ndimage.filters import convolve as filter2
import modules_baseline as mod


######################################################################
######################################################################
afterImg = cv.imread('image2.png', 0).astype(float)
beforeImg = cv.imread('image1.png', 0).astype(float)

print('image0 shape',afterImg.shape)
l=mod.determine_numLevels(afterImg,1/0.8)
print(l)
u,v = mod.Hs_pyram(beforeImg, afterImg,factor=1/0.8, alpha = 15, delta = 10**-1,Level=l,itmax=900,kernel_size=5,downsampling_gauss=0.80)
mod.draw_quiver(u, v, beforeImg)



