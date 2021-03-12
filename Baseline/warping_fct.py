import numpy as  np 
import cv2
from numpy.lib.function_base import meshgrid 
from scipy.ndimage.filters import convolve as filter2
from scipy.ndimage import  median_filter
import matplotlib.pyplot as plt
import math 
#############################################
def Expand(image,factor):
    ''' Expand image by facor times'''
    N, M = image.shape
    newN = int(N * factor)
    newM = int(M * factor)
    newImage =cv2.resize(image, (newM,newN), interpolation = cv2.INTER_AREA).astype(float)
    return newImage
#############################################
def Extract(Image,factor):
    N, M = Image.shape
    newN = int(N/ factor)
    newM = int(M / factor)
    newImage =cv2.resize(Image, (newM,newN), interpolation = cv2.INTER_AREA ).astype(float)
    return newImage
#############################################
def warp_image2(Image,XI,YI,h):
 
    # We add the flow estimated to the second image coordinates, remap them towards the ogriginal image  and finally  calculate the derivatives of the warped image
    Image=np.array(Image,np.float32)
    XI=np.array(XI,np.float32)
    YI=np.array(YI,np.float32)
    WImage=cv2.remap(Image,XI,YI,interpolation=cv2.INTER_CUBIC)
    Ix=filter2(WImage, h)
    Iy=filter2(WImage, h.T)
    
    #Iy=cv2.remap(Iy,XI,YI,interpolation=cv2.INTER_LINEAR)   
    #Ix=cv2.remap(Ix,XI,YI,interpolation=cv2.INTER_LINEAR)
    return [WImage,Ix,Iy]
    
############################################
def derivatives(Image1,Image2,u,v,h,b):
    N,M=Image1.shape
    #x = np.array(range(N))
    #y = np.array(range(M))
    y=np.linspace(0,N-1,N)
    x=np.linspace(0,M-1,M)
    x,y=np.meshgrid(x,y)

    x=x+u; y=y+v
    WImage,Ix,Iy=warp_image2(Image2,x,y,h)  # Derivatives of the secnd image 

    It= WImage-Image1 # Temporal deriv
    I1x=filter2(Image1, h) # spatial derivatives for the first image 
    I1y=filter2(Image1, h.T)
    
    Ix  = b*Ix+(1-b)*I1x # Averaging 
    Iy  = b*Iy+(1-b)*I1y

    It=np.nan_to_num(It) #Remove Nan values on the derivatives 
    Ix=np.nan_to_num(Ix)
    Iy=np.nan_to_num(Iy)
    out_bound= np.where((y > N-1) | (y<0) | (x> M-1) | (x<0))
    Ix[out_bound]=0 # setting derivatives value on out of bound pixels to 0  
    Iy[out_bound]=0
    It[out_bound]=0
    return [Ix,Iy,It]
################################################################################
def warping_step(beforeImg, afterImg, alpha, delta,u,v,kernel_size,downsampling_gauss,h,b,factor):
    #removing noise
    deviation_gausse=float(1/math.sqrt(2*downsampling_gauss))
    beforeImg  = cv2.GaussianBlur(beforeImg, ksize=(kernel_size, kernel_size), sigmaX=deviation_gausse,sigmaY=0)
    afterImg = cv2.GaussianBlur(afterImg,ksize=(kernel_size, kernel_size), sigmaX=deviation_gausse,sigmaY=0)
    
    # set up initial values

    fx, fy, ft = derivatives(beforeImg,afterImg,u,v,h,b)

    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    for iter in range(10):
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        d =  alpha + fx**2 + fy**2
        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)
    u=Expand(u,factor)
    v=Expand(v,factor)
    u=median_filter(u,size=5)
    v=median_filter(v,size=5)
    return [u, v]
###################################################################################
def determine_numLevels(image,factor):
    # factor must be >1 
    a=np.min(image.shape)
    lev=math.log((a/25))/math.log(factor)
    return int(lev)
######################################################################################
def warping_pyram(beforImage,afterImage,factor,alpha,delta,Level,kernel_size,downsampling_gauss,h,b):
    for lev in range(Level-1,-1,-1):

        if lev==(Level-1):
            image0=Extract(beforImage,factor**lev)
            image1=Extract(afterImage,factor**lev)
            u0 = np.zeros((image0.shape[0], image0.shape[1]))
            v0 = np.zeros((image0.shape[0], image0.shape[1]))
            print('shape of u0',u0.shape)
        print('debut lev',lev)
        u,v=warping_step(image0,image1, alpha, delta,u0,v0,kernel_size,downsampling_gauss,h,b,factor)
        #draw_quiver(u, v, beforeImg)
        print('fin lev',lev)
        (local_N,local_M)=image1.shape
        if (lev !=0):
            u0=factor*cv2.resize(u, (int(local_M*factor),int(local_N*factor)), interpolation = cv2.INTER_CUBIC)
            v0=factor*cv2.resize(v, (int(local_M*factor),int(local_N*factor)), interpolation = cv2.INTER_CUBIC)
            image0=Expand(image0,factor)
            image1=Expand(image1,factor)
        #print(u.shape)
    return[u,v]



#####################################""
def get_magnitude(u, v):
    scale = 3
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 10):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg

def draw_quiver(u,v,beforeImg):
    scale = 3
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'red')

    plt.draw()
    plt.show()





'''(sy,sx)=Image.shape
XI=np.reshape(XI,(1,sx*sy))
YI=np.reshape(YI,(1,sx*sy))
fXI = np.floor(XI);
cXI = fXI + 1;
fYI = np.floor(YI);
cYI = fYI + 1;
indx = (fXI<0) | (cXI>sx-1) | (fYI<0) | (cYI>sy-1);
fXI = max(0, min(sx-1, fXI));
cXI = max(0, min(sx-1, cXI));
fYI = max(0, min(sy-1, fYI));
cYI = max(0, min(sy-1, cYI));'''