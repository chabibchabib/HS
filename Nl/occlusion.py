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
from skimage import feature
import weighted_median as wm
##################################################################################
def detect_occlusion(Image1,Image2,u,v,h,b,sigma_d,sigma_i):
    div=np.gradient(u,axis=1)+np.gradient(v,axis=0)
    div[div>0]=0
    [Ix,Iy,It]=fo.derivatives(Image1,Image2,u,v,h,b)
    occ = np.exp(-div**2/2/sigma_d**2)*np.exp(-It**2/2/ sigma_i**2)
    return occ
####################################################################################
def sub2ind(sz,row,col) :
    res=[] 
    for i in range(row.shape[0]): 
        res.append(row[i]+(col[i]-1)*sz[0]) 
    return res
####################################################################################
def denoise_color_weighted_medfilt2(u,v, im1,im2, occ, bfhsz, mfsz, sigma_i, fullVersion):
    sigma_x = 7   #  spatial distance (7)
    dilate_sz = (5, 5)  #dilation window size for flow edge region [5 5]
    sz = im1.shape
    uo= median_filter(u, mfsz, mode='mirror')##
    vo=median_filter(v, mfsz, mode='mirror') ##
    e1=feature.canny(u)                        #detect edge                           
    e2=feature.canny(v)                                                  
    e=np.logical_or(e1,e2) 
    mask = cv2.dilate(np.array(e,dtype=np.float32), np.ones(dilate_sz) ) ## #dilate area 
    if fullVersion==True:
        mask = np.ones(mask.shape)
    
    [indx_row, indx_col] = np.where(mask ==1)

    pad_u=np.pad(u,((bfhsz,bfhsz),(bfhsz,bfhsz)),mode='symmetric')
    pad_v  =np.pad(v,((bfhsz,bfhsz),(bfhsz,bfhsz)),mode='symmetric')     
    pad_im1 = np.pad(im1, ((bfhsz,bfhsz),(bfhsz,bfhsz)),mode='symmetric')      
    pad_im2 = np.pad(im2, ((bfhsz,bfhsz),(bfhsz,bfhsz)),mode='symmetric')      

    pad_occ= np.pad(occ, ((bfhsz,bfhsz),(bfhsz,bfhsz)),mode='symmetric')

    (H ,W) = pad_u.shape
    #print('H',H)
    # Divide into several groups for memory reasons ~70,000 causes out of memory
    Indx_Row = indx_row
    Indx_Col = indx_col
    N        = len(Indx_Row) # number of elements to process
    n        = 4e4           # number of elements per batch
    nB       = math.ceil(N/n)
    
    for ib in range(nB):
        istart = int((ib)*n)
        iend   = int(min((ib+1)*n, N))
        #print( istart,iend)
        indx_row = Indx_Row[istart:iend]
        indx_col = Indx_Col[istart:iend]
        '''print('indx_row', indx_row)
        print('indx_col', indx_col)'''
        [C,R]=np.meshgrid(range(-bfhsz,bfhsz+1),range(-bfhsz,bfhsz+1))

        nindx = R + C*H
        
        cindx = indx_row+1 +bfhsz+ (indx_col+bfhsz)*H
        
        cindx=np.array([cindx])
        #print('dnix',nindx)
        #print('cdnix',cindx)
        
        pad_indx =np.matlib.repmat(np.reshape(nindx ,(nindx.shape[0]*nindx.shape[1],1),'F'),1,len(indx_row) )+np.matlib.repmat(np.reshape(cindx ,(cindx.shape[0]*cindx.shape[1],1),'F').T,(bfhsz*2+1)**2,1 )-1
        #print('pad idx',pad_indx)
        # spatial weight
        tmp = np.exp(-(C**2 + R**2) /2/sigma_x**2 )
        weights = np.matlib.repmat(np.reshape(tmp ,(tmp.shape[0]*tmp.shape[1],1),'F'), 1, len(indx_row)) 
        #print(weights.shape)
        tmp_w = np.zeros(weights.shape)
        
        tmp1 = pad_im1
        #print(tmp.shape)
        J=np.array(pad_indx/tmp1.shape[0],dtype=np.int)
        I=np.array(pad_indx-(J)*tmp1.shape[0],dtype=np.int)
        J_cindx=np.array(cindx/tmp1.shape[0],dtype=np.int)
        I_cindx=np.array(cindx-(J_cindx)*tmp1.shape[0],dtype=np.int)
        '''print('J',J_cindx)
        print('I',I_cindx)
        print('cindx ',cindx)
        print(tmp_w.shape)'''
        #tmp_w = tmp_w + ( tmp1[I,J]+np.matlib.repmat( tmp1[ np.reshape(cindx,(cindx.shape[0]*cindx.shape[1],1),'F')   ].T, (bfhsz*2+1)^2, 1) )**2
        tmp_w = tmp_w + ( tmp1[I,J]+np.matlib.repmat( tmp1[  I_cindx,  J_cindx  ], (bfhsz*2+1)**2, 1) )**2

        
        tmp1 = pad_im2
        #tmp_w = tmp_w + ( tmp1[pad_indx] +np.matlib.repmat( tmp1[I_cindx,  J_cindx ].T, (bfhsz*2+1)^2, 1) )**2
        tmp_w = tmp_w + ( tmp1[I,J]+np.matlib.repmat( tmp1[  I_cindx,  J_cindx  ], (bfhsz*2+1)**2, 1) )**2

        tmp_w = tmp_w/2

        weights = weights* np.exp(-tmp_w/2/sigma_i**2)

         #Occlusion weight    
        weights = weights*pad_occ[I,J]
        #Normaliser
        weights=weights/np.matlib.repmat( np.sum(weights,0),(bfhsz*2+1)**2, 1  )

        neighbors_u = pad_u[I,J]
        neighbors_v = pad_v[I,J]
        #print('neib',neighbors_v.shape)

        '''# weighted average as initial value
        u_w = np.sum(weights*neighbors_u,0)
        v_w = np.sum(weights*neighbors_v,0)'''

        u_w=wm.weighted_median(weights, neighbors_u)
        v_w= wm.weighted_median(weights, neighbors_v)
        #print(u.shape)
        uo[indx_row,indx_col]=u_w
        vo[indx_row,indx_col]=v_w
        #print(uo.shape)
    return [uo,vo]

        

#######################################################################################################
#N=10; M=11
h=np.array([[1 ,-8 ,0 ,8 ,-1]])/12
b=0.5
sigma_d=7 
sigma_i=7
fullVersion=True
bfhsz=10
mfsz=(5,5)
#im1=np.floor(10*np.random.rand(N,M),dtype=np.float32)
#im2=np.floor(10*np.random.rand(N,M),dtype=np.float32)
im1=cv2.imread('Im1.png',0)
im2=cv2.imread('Im2.png',0)
im1=np.array(im1,dtype=np.float32)
im2=np.array(im2,dtype=np.float32)

(N,M)=im1.shape
print(N,M)
u=np.zeros((N,M))
v=np.zeros((N,M))

occ=detect_occlusion(im1,im2,u,v,h,b,0.3,20)
print(occ)
[uo,vo]=denoise_color_weighted_medfilt2(u,v, im1,im2, occ, bfhsz, mfsz, sigma_i, fullVersion)
print(vo)
        



        


