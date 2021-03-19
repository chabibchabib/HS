import numpy as np 
import scipy.ndimage
import cv2
#import math
#from scipy.ndimage.filters import convolve 
##################################
def Extract(Image,factor):
    N, M = Image.shape
    newN = round(N/ factor+0.001)
    newM = round(M / factor+0.001)
    newImage =cv2.resize(Image, (newM,newN), interpolation = cv2.INTER_AREA ).astype(float)
    return newImage

def determine_numLevels(image,factor):
    # factor must be >1 
    a=np.max(image.shape)
    lev=math.log((a/30))/math.log(factor)
    return int(lev)

def lorentzian(x,sigma):
    res=np.log(np.ones(x.shape)+(x**2*1/(2*sigma**2)))
    return res

def quadratic(x,sigma):
    res=x**2
    return res
########################################
def warp_image2(Image,XI,YI,h):
 
    # We add the flow estimated to the second image coordinates, remap them towards the ogriginal image  and finally  calculate the derivatives of the warped image
    Image=np.array(Image,np.float32)
    XI=np.array(XI,np.float32)
    YI=np.array(YI,np.float32)
    WImage=cv2.remap(Image,XI,YI,interpolation=cv2.INTER_CUBIC)
    Ix=convolve(WImage, h)
    Iy=convolve(WImage, h.T)
    
    Iy=cv2.remap(Iy,XI,YI,interpolation=cv2.INTER_CUBIC)   
    Ix=cv2.remap(Ix,XI,YI,interpolation=cv2.INTER_CUBIC)
    return [WImage,Ix,Iy]
    
############################################
def derivatives(Image1,Image2,u,v,h,b):
    N,M=Image1.shape
    y=np.linspace(0,N-1,N)
    x=np.linspace(0,M-1,M)
    x,y=np.meshgrid(x,y)

    x=x+u; y=y+v
    WImage,I2x,I2y=warp_image2(Image2,x,y,h)  # Derivatives of the secnd image 
    It= WImage-Image1 # Temporal deriv
    
    I1x=convolve(Image1, h) # spatial derivatives for the first image 
    I1y=convolve(Image1, h.T)
    
    Ix  = b*I2x+(1-b)*I1x # Averaging 
    Iy  = b*I2y+(1-b)*I1y


    It=np.nan_to_num(It) #Remove Nan values on the derivatives 
    Ix=np.nan_to_num(Ix)
    Iy=np.nan_to_num(Iy)
    out_bound= np.where((y > N-1) | (y<0) | (x> M-1) | (x<0))
    Ix[out_bound]=0 # setting derivatives value on out of bound pixels to 0  
    Iy[out_bound]=0
    It[out_bound]=0
    return [Ix,Iy,It]
########################################
def flow_operator(image1,image2):


########################################
def compute_flow_base(Image1,Image2,u,v,rho_s,rho_d,h,b):
    ta=rho_d/rho_s
    Ix,Iy,Iz=derivatives(Image1,Image2,u,v,h,b)



