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
##############################################################
def compute_image_pyram(Im1,Im2,ratio,N_levels,gaussian_sigma):
    P1=[]
    P2=[]
    tmp1=ri.scale_image(Im1,0,255)
    tmp2=ri.scale_image(Im2,0,255)
    P1.append(tmp1)
    P2.append(tmp2)

    for lev in range(1,N_levels):
        tmp1=gaussian_filter(tmp1,gaussian_sigma)
        tmp2=gaussian_filter(tmp2,gaussian_sigma)
        sz=np.round(np.array(tmp1.shape,dtype=np.float32)*ratio)
        #tmp1=resize(tmp1,(sz[0],sz[1]),anti_aliasing=True)
        #tmp2=resize(tmp2,(sz[0],sz[1]),anti_aliasing=True)
        tmp1=resize(tmp1,(sz[0],sz[1]),anti_aliasing=False,mode='symmetric')
        tmp2=resize(tmp2,(sz[0],sz[1]),anti_aliasing=False,mode='symmetric')

        P1.append(tmp1)
        P2.append(tmp2)
    return [P1,P2]


def resample_flow_unequal(u,v,sz,ordre_inter):
    osz=u.shape
    ratioU=sz[0]/osz[0]
    ratioV=sz[1]/osz[1]
    u=resize(u,sz,order=ordre_inter)*ratioU
    v=resize(v,sz,order=ordre_inter)*ratioV
    return u,v


############################################################
def compute_flow(Im1,Im2,u,v,iter_gnc,gnc_pyram_levels,gnc_factor,gnc_spacing, pyram_levels,factor,spacing,ordre_inter, alpha,lmbda, size_median_filter,h,coef,S,max_linear_iter,max_iter):
    Im1,Imm1=ri.decompo_texture(Im1, 1/8, 100, 0.95, False)
    Im2,Imm1=ri.decompo_texture(Im2, 1/8, 100, 0.95, False)

    P1,P2=compute_image_pyram(Im1,Im2,1/factor,pyram_levels,math.sqrt(spacing)/math.sqrt(2))
    P1_gnc,P2_gnc=compute_image_pyram(Im1,Im2,1/gnc_factor,gnc_pyram_levels,math.sqrt(gnc_spacing)/math.sqrt(2))
    #print(P2[0][:5,:5])
    for i in range(iter_gnc):
        if i==0:
            py_lev=pyram_levels
        else:
            py_lev=gnc_pyram_levels
        print('pylev',py_lev)
        for lev in range(py_lev-1,-1,-1):
            if i==0:
                Image1=P1[lev]; Image2=P2[lev]
                sz= Image1.shape
            else:
                
                Image1=P1_gnc[lev]; Image2=P2_gnc[lev]
                sz= Image1.shape
            
            u,v=resample_flow_unequal(u,v,sz,ordre_inter)

            if (lev==0) and (i==iter_gnc) :    #&& this.noMFlastlevel
                median_filter_size =0
            else: 
                median_filter_size=size_median_filter
                    
            ''' Run flow method on subsampled images
          if small.useInlinePCG
              uv = compute_flow_base_pcg_inline(small, uv);
          else'''
            print('small images 1')
            print(Image1[0:5,0:5])
            print('small images 2')
            print(Image2.shape)
            u,v=fo.compute_flow_base(Image1,Image2,max_iter,max_linear_iter,u,v,alpha,lmbda,S,median_filter_size,h,coef)
            print('u')
            print(u[0:5,0:5])
            print('v')
            print(v[0:5,0:5])

            if iter_gnc > 0:

                new_alpha  = 1 - (i+1)/ (iter_gnc)
                alpha = min(alpha, new_alpha)
                alpha = max(0,alpha)

    return u,v

            

            







'''Im1=np.loadtxt('Im1.txt',dtype=np.float32)
Im2=np.loadtxt('Im2.txt',dtype=np.float32)'''
Im1=cv2.imread('Im1.png',0)
Im2=cv2.imread('Im2.png',0)
Im1=np.array(Im1,dtype=np.float32)
Im2=np.array(Im2,dtype=np.float32)

u=np.zeros((Im1.shape)); v=np.zeros((Im1.shape))
iter_gnc=3
gnc_pyram_levels=2
gnc_factor=1.25
gnc_spacing=1.25

pyram_levels=1
factor=2
spacing=2
ordre_inter=1
alpha=1
lmbda=0.06
size_median_filter=5
h=np.array([[-1 ,8, 0 ,-8 ,1 ]]); h=h/12
coef=0.5
S=[]
S.append(np.array([[1,-1]]))
S.append(np.array([[1],[-1]]))
max_linear_iter=1
max_iter=10


'''mat=np.matrix(Im1)
with open('Im1.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.2f')

mat=np.matrix(Im2)
with open('Im2.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.2f')'''
pyram_levels=ri.compute_auto_pyramd_levels(Im1,spacing)
u,v=compute_flow(Im1,Im2,u,v,iter_gnc,gnc_pyram_levels,gnc_factor,gnc_spacing, pyram_levels,factor,spacing,ordre_inter,
 alpha,lmbda, size_median_filter,h,coef,S,max_linear_iter,max_iter)

print('Im1')
print(Im1[0:5,0:5])

print('Im2')
print(Im2[0:5,0:5])

print('u')
print(u[0:5,0:5])

print('v')
print(v[0:5,0:5])
N,M=Im1.shape
y=np.linspace(0,N-1,N)
x=np.linspace(0,M-1,M)
x,y=np.meshgrid(x,y)
x2=x+u; y2=y+v
x2=np.array(x2,dtype=np.float32)
y2=np.array(y2,dtype=np.float32)
I=cv2.remap(np.array(Im1,dtype=np.float32),x2,y2,cv2.INTER_LINEAR)
norme=np.linalg.norm(I-Im2)/np.linalg.norm(Im2)
print(norme)
cv2.imwrite('I_first.png',I)

print(I[0:5,0:5])
