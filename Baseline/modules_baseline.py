import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2 as cv
import math
from scipy.ndimage.filters import convolve as filter2

def gaussian(sigma,x,y):
    ''' Gaussian Distribution function  '''
    a= 1/(np.sqrt(2*np.pi)*sigma)
    b=math.exp(-(x**2+y**2)/(2*(sigma**2)))
    c = a*b
    return a*b

def gaussian_kernel(sigma ):
    ''' Compute Gaussian Kernel '''
    G=np.zeros((5,5))
    for i in range(-2,3):
        for j in range(-2,3):
            G[i+1,j+1]=gaussian(sigma,i,j)
    return G


def Hs_Expand(image,factor):
    ''' Expand image by facor times'''
    N, M = image.shape
    newN = int(N * factor)
    newM = int(M * factor)
    newImage =cv.resize(image, (newM,newN), interpolation = cv.INTER_LINEAR).astype(float)
    return newImage


def Hs_extract(Image,factor):
    N, M = Image.shape
    newN = int(N/ factor)
    newM = int(M / factor)
    newImage =cv.resize(Image, (newM,newN), interpolation = cv.INTER_LINEAR ).astype(float)
    return newImage
#############################################################################
def image_derivatives(img1, img2):
    # Computing Image derivatives 
    #Gx = np.array([[-1, 1], [-1, 1]]) * 0.25
    Gx=float(1/12)* np.array([-1 8 0 -8 1])
    Gy = np.array([[-1, -1], [1, 1]]) * 0.25
    Gt = np.ones((2, 2)) * 0.25
    fx = filter2(img1,Gx) + filter2(img2,Gx)
    fy = filter2(img1, Gy) + filter2(img2, Gy)
    ft = filter2(img1, -Gt) + filter2(img2, Gt)
    return [fx,fy, ft]


def computeHS(beforeImg, afterImg, alpha, delta,itmax,u,v,kernel_size,downsampling_gauss):
    #removing noise
    deviation_gausse=float(1/math.sqrt(2*downsampling_gauss))
    beforeImg  = cv.GaussianBlur(beforeImg, ksize=(kernel_size, kernel_size), sigmaX=deviation_gausse,sigmaY=0)
    afterImg = cv.GaussianBlur(afterImg,ksize=(kernel_size, kernel_size), sigmaX=deviation_gausse,sigmaY=0)

    # set up initial values

    fx, fy, ft = image_derivatives(beforeImg, afterImg)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    iter_counter = 0
    while True:
        iter_counter += 1
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        d = 4 * alpha**2 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        #print(iter_counter)
        if  diff < delta or iter_counter > itmax:
            print("iteration number: ", iter_counter)
            print('erreur=',diff)
            
            break
    #draw_quiver(u, v, beforeImg)
    return [u, v]
###################################################################
def Hs_pyram(beforImage,afterImage,factor,alpha,delta,Level,itmax,kernel_size,downsampling_gauss):
    for lev in range(Level-1,-1,-1):

        if lev==(Level-1):
            image0=Hs_extract(beforImage,factor**lev)
            image1=Hs_extract(afterImage,factor**lev)
            u0 = np.zeros((image0.shape[0], image0.shape[1]))
            v0 = np.zeros((image0.shape[0], image0.shape[1]))
            print('shape of u0',u0.shape)
        print('debut lev',lev)
        u,v=computeHS(image0,image1, alpha, delta,itmax,u0,v0,kernel_size,downsampling_gauss)
        #draw_quiver(u, v, beforeImg)
        print('fin lev',lev)
        (local_N,local_M)=image1.shape
        #if (lev !=0):
        u0=factor*cv.resize(u, (int(local_M*factor),int(local_N*factor)), interpolation = cv.INTER_LINEAR)
        v0=factor*cv.resize(v, (int(local_M*factor),int(local_N*factor)), interpolation = cv.INTER_LINEAR)
        image0=Hs_Expand(image0,factor)
        image1=Hs_Expand(image1,factor)
        #print(u.shape)
    return[u,v]
######################################################################
#plot
def get_magnitude(u, v):
    scale = 3
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
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

######################################################
def determine_numLevels(image,factor):
    # factor must be >1 
    a=np.min(image.shape)
    lev=math.log((a/25))/math.log(factor)
    return int(lev)
