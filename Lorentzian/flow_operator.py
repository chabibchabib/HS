import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt

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
    ''' Using Lorentzian function '''
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
    if( ((np.max(pp_su)-np.max(pp_su)<1E-06)   ) and ((np.max(pp_sv)-np.max(pp_sv)<1E-06)   ) and ((np.max(pp_d)-np.max(pp_d)<1E-06)   ) ):
        iterative=False
    else:
        iterative=True
    return [A,b,iterative]
#############################################################
def flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda):
    '''using quadratic function ''' 
    #sz=(Ix.shape[0],Ix.shape[1])
    sz=np.shape(Ix)
    npixels=Ix.shape[1]*Ix.shape[0]
    print(npixels)
    FU=np.zeros((npixels,npixels))
    FV=np.zeros((npixels,npixels))
    quadr_ov_x=np.vectorize(deriv_quadra_over_x)
    for i in range(len(S)):
        M=conv_matrix(S[i],sz)
        #print(M)
        u_=np.matmul(M,np.reshape((u+du),(npixels,1),'F'))
        v_=np.matmul(M,np.reshape((v+dv),(npixels,1),'F'))
        #pp_su=deriv_quadra_over_x(u_,1)
        #pp_sv=deriv_quadra_over_x(v_,1)
        pp_su=quadr_ov_x(u_,1)
        pp_sv=quadr_ov_x(v_,1)
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
    
    pp_d=deriv_quadra_over_x(np.reshape(It,(npixels,1),'F'),(1.5/0.03))
    
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
    if( ((np.max(pp_su)-np.max(pp_su)<1E-06)   ) and ((np.max(pp_sv)-np.max(pp_sv)<1E-06)   ) and ((np.max(pp_d)-np.max(pp_d)<1E-06)   ) ):
        iterative=False
    else:
        iterative=True
    return [A,b,iterative]
#############################################################
def  compute_flow_base(max_iter,max_linear_iter,u,v,alpha,lmbda,S,size_median_filter):
    for i in range(max_iter):
        du=np.zeros((u.shape)); dv=np.zeros((v.shape))
        npixels=u.shape[0]*u.shape[1]
        #[It Ix Iy] = partial_deriv(this.images, uv
        for j in range(max_linear_iter):
            if (alpha==1):
                #[A,b,iterative]=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda)
                [A,b,iterative]=flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda)
            elif((alpha>0)  and (alpha !=1) ):
                [A,b,iterative]=flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda)
                [A1,b1,iterative1]=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda)
                
                A=alpha*A+(1-alpha)*A1
                b=alpha*b+(1-alpha)*b1
            elif(alpha==0):
                [A,b,iterative]=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda)
            
            x=np.matmul(np.linalg.inv(A),b)
            du=np.reshape(x[0:npixels], (u.shape[0],u.shape[1]),'F' )
            dv=np.reshape(x[npixels:2*npixels], (u.shape[0],u.shape[1]),'F' )
            (u0,v0)=(u,v)
            (u,v)=(u+du,v+dv)

            u=medfilt(u,size_median_filter)
            v=medfilt(v,size_median_filter)
            du = u- u0
            dv = v- v0
            u= u0
            v=v0

        u = u + du
        v = v + dv
    return [u,v]


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
#A,b,iterative=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda)
#print('A',A[0,:10])
#print(iterative)
compute_flow_base(10,1,u,v,0.5,lmbda,S,5)