import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
#from scipy.ndimage.filters import convolve 
from scipy.sparse import spdiags

def conv_matrix(F,sz):
    '''Fshape=np.array(F.shape)
    size=sz+Fshape-1
    size=np.array(size,np.int)
    valid=np.ones((size[0,0],size[0,1]))
    pad_valid = np.ones(sz+F.shape-1);
    if(F.shape==(1,2)):

        valid[:,0]=0;
        valid[:,size[0,1]-1]=0
        pad_valid[:,2]=0
    elif(F.shape==(2,1)):
        valid[0,:]=0;
        valid[size[0,0]-1,:]=0
        pad_valid[6,:]=0'''
    M=np.zeros((sz[0]*sz[1],sz[0]*sz[1])) 
    if( F.shape==(1,2)):
        for i in range(sz[0],sz[0]*sz[1]):      
            M[i,i-sz[0]]=-1;      
        for i in range(sz[0],sz[0]*sz[1]): 
            M[i,i]=1
    elif(F.shape==(2,1)):
        for i in range(sz[0]*sz[1]):
            if(i==0):
                M[i,i]=0
            else:
                M[i,i]=1
                M[i,i-1]=-1
    return M
########################################################
def deriv_lorentz_over_x(x,sigma):
     y = 2 / (2 * sigma**2 + x**2);
     return y
def deriv_quadra_over_x(x,sigma):
    y = 2 / (sigma**2);
    return y
########################################################
def flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda):
    #sz=(Ix.shape[0],Ix.shape[1])
    sz=np.shape(Ix)
    npixels=Ix.shape[1]*Ix.shape[0]
    print(npixels)
    FU=np.zeros((npixels,npixels))
    FV=np.zeros((npixels,npixels))
    for i in range(len(S)):
        M=conv_matrix(S[i],sz)
        #print(M)
        u_=np.matmul(M,np.reshape((u+du),(npixels,1),'F'))
        v_=np.matmul(M,np.reshape((v+dv),(npixels,1),'F'))
        pp_su=deriv_lorentz_over_x(u_,0.03)
        pp_sv=deriv_lorentz_over_x(v_,0.03)
        #print(pp_sv)
        
        FU        = FU+ np.matmul(M.T,np.matmul(spdiags(pp_su.T, 0, npixels, npixels).toarray(),M));
        FV        = FV+ np.matmul(M.T,np.matmul(spdiags(pp_sv.T, 0, npixels, npixels).toarray(),M));
    
    MM = np.vstack( (np.hstack ( ( -FU, np.zeros((npixels,npixels)) ) )  ,  np.hstack( ( np.zeros((npixels,npixels)) , -FV ) )  ))  
    #print('MM')
    #print(MM)
    Ix2 = Ix*Ix
    Iy2 = Iy*Iy
    Ixy = Ix*Iy
    Itx = It*Ix
    Ity = It*Iy

    It = It + Ix*du+ Iy*dv;
    
    pp_d=deriv_lorentz_over_x(np.reshape(It,(npixels,1),'F'),1.5)
    
    tmp=pp_d*np.reshape(Ix2,(npixels,1),'F')
    duu = spdiags(tmp.T, 0, npixels, npixels).toarray()
    
    tmp = pp_d*np.reshape(Iy2,(npixels,1),'F')
    
    dvv = spdiags(tmp.T, 0, npixels, npixels).toarray()
    
    tmp = pp_d*np.reshape(Ixy,(npixels,1),'F')
    
    dduv = spdiags(tmp.T, 0, npixels, npixels).toarray()

    A = np.vstack( (np.hstack ( ( duu, dduv ) )  ,  np.hstack( ( dduv , dvv ) )  )) - lmbda*MM
    b=np.matmul(lmbda*MM, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )) - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) ) 
    #print(b[:10])
    #print(np.matmul(lmbda*MM, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )))
    return [A,b]


#############################################################
#Ix=np.array(10*np.random.rand(5,3),np.int)
#Iy=2*Ix; It=3*Ix;
#u=Ix; v=Iy; du=Ix/2; dv=Iy/4; lmbda=1; image1=np.zeros((Ix.shape)); image2=np.zeros((Ix.shape));
S=[];
S.append(np.array([[1,-1]]))
S.append(np.array([[1],[-1]]))

It =np.array([[-3.69547,   0.41807  , 2.02836 , -0.53392 ,  0.00000],
  [ 1.63937 , -1.49628 , -2.13478  , 0.14007 ,  0.00000],
  [-0.99281 , -0.62484 , -1.55485 ,  0.38739 ,  0.00000],
  [-0.19625 , -1.58552 ,  0.51390 , -0.05108 ,  0.00000],
  [ 0.00000,   0.00000 ,  0.00000 ,  0.00000,   0.00000]])

Ix = np.array([[ 1.66114 ,  1.46993 , -0.59511  , 0.65337 ,  0.00000],
  [-1.06698  ,-0.46049 ,  0.94558  , 0.91834  , 0.00000],
   [0.86646,   0.20117  ,-0.89750  , 0.28706 ,  0.00000],
  [-0.21850  , 0.66434  , 1.01487  ,-0.38705  , 0.00000],
  [ 0.00000 ,  0.00000  , 0.00000 ,  0.00000  , 0.00000]])

Iy = np.array([[ 1.50229  ,-1.37956  ,-0.67601 ,  0.18091  , 0.00000],
  [ 1.21001 ,  0.56375 , -0.12846 , -0.15330   ,0.00000],
  [-0.80391 ,  0.31335 ,  0.45995 , -0.00943  , 0.00000],
 [ -0.18017 , -1.22093  , 0.84927  , 2.06008   ,0.00000],
  [ 0.00000  , 0.00000  , 0.00000 ,  0.00000  , 0.00000]])
u=np.zeros((Ix.shape))
v=np.zeros((Ix.shape))
du=np.zeros((u.shape))
dv=np.zeros((u.shape))
lmbda=1
print('shape',u.shape)
A,b=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda)
#print('A',A[0,:10])
print('b',b)