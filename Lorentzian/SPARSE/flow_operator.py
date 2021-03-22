import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import scipy.sparse as sparse
###########################################################
def warp_image2(Image,XI,YI,h):
 
    # We add the flow estimated to the second image coordinates, remap them towards the ogriginal image  and finally  calculate the derivatives of the warped image
    Image=np.array(Image,np.float32)
    XI=np.array(XI,np.float32)
    YI=np.array(YI,np.float32)
    WImage=cv2.remap(Image,XI,YI,interpolation=cv2.INTER_CUBIC)
    Ix=filter2(WImage, h)
    Iy=filter2(WImage, h.T)
    
    Iy=cv2.remap(Iy,XI,YI,interpolation=cv2.INTER_CUBIC)   
    Ix=cv2.remap(Ix,XI,YI,interpolation=cv2.INTER_CUBIC)
    return [WImage,Ix,Iy]
    
############################################
def derivatives(Image1,Image2,u,v,h,b):
    N,M=Image1.shape
    #x = np.array(range(N))
    #y = np.array(range(M))
    y=np.linspace(0,N-1,N)
    x=np.linspace(0,M-1,M)
    x,y=np.meshgrid(x,y)
    Ix=np.zeros((N,M))
    Iy=np.zeros((N,M))
    x=x+u; y=y+v
    WImage,I2x,I2y=warp_image2(Image2,x,y,h)  # Derivatives of the secnd image 
    '''Gt = np.ones((2, 2)) * 0.25
    It = filter2(Image1, -Gt) + filter2(WImage, Gt)'''
    It= WImage-Image1 # Temporal deriv
    
    Ix=filter2(Image1, h) # spatial derivatives for the first image 
    Iy=filter2(Image1, h.T)
    #print("shapes",np.array(0.5*Ix).shape)
    #print('bshape',b)
    Ix  = b*I2x+(1-b)*Ix           # Averaging 
    Iy  = b*I2y+(1-b)*Iy


    It=np.nan_to_num(It) #Remove Nan values on the derivatives 
    Ix=np.nan_to_num(Ix)
    Iy=np.nan_to_num(Iy)
    out_bound= np.where((y > N-1) | (y<0) | (x> M-1) | (x<0))
    Ix[out_bound]=0 # setting derivatives value on out of bound pixels to 0  
    Iy[out_bound]=0
    It[out_bound]=0
    return [Ix,Iy,It]
############################################################
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
    #M=np.zeros((sz[0]*sz[1],sz[0]*sz[1])) 
    M=sparse.csr_matrix( ( sz[0]*sz[1],sz[0]*sz[1] ),dtype=np.float32)
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
    #print(npixels)
    #FU=np.zeros((npixels,npixels))
    #FV=np.zeros((npixels,npixels))
    FU=sparse.csr_matrix((npixels,npixels),dtype=np.float32)
    FV=sparse.csr_matrix((npixels,npixels),dtype=np.float32)
    for i in range(len(S)):
        M=conv_matrix(S[i],sz)
        #print(M)
        '''u_=np.matmul(M,np.reshape((u+du),(npixels,1),'F'))
        v_=np.matmul(M,np.reshape((v+dv),(npixels,1),'F'))'''
        u_=sparse.csr_matrix.dot(M,np.reshape((u+du),(npixels,1),'F'))
        v_=sparse.csr_matrix.dot(M,np.reshape((v+dv),(npixels,1),'F'))
        pp_su=deriv_lorentz_over_x(u_,0.03)
        pp_sv=deriv_lorentz_over_x(v_,0.03)
        #print(pp_sv)
        
        '''FU        = FU+ np.matmul(M.T,np.matmul(spdiags(pp_su.T, 0, npixels, npixels).toarray(),M));
        FV        = FV+ np.matmul(M.T,np.matmul(spdiags(pp_sv.T, 0, npixels, npixels).toarray(),M));'''
        FU        = FU+ sparse.csr_matrix.dot(M.T,sparse.csr_matrix.dot(spdiags(pp_su.T, 0, npixels, npixels),M));
        FV        = FV+ sparse.csr_matrix.dot(M.T,sparse.csr_matrix.dot(spdiags(pp_sv.T, 0, npixels, npixels),M));
    #MM = np.vstack( (np.hstack ( ( -FU, np.zeros((npixels,npixels)) ) )  ,  np.hstack( ( np.zeros((npixels,npixels)) , -FV ) )  ))  
    MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.csr_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.csr_matrix((npixels,npixels)) , -FV ) )  ))  

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
    duu = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Iy2,(npixels,1),'F')
    
    dvv = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Ixy,(npixels,1),'F')
    
    dduv = spdiags(tmp.T, 0, npixels, npixels)

    '''A = np.vstack( (np.hstack ( ( duu, dduv ) )  ,  np.hstack( ( dduv , dvv ) )  )) - lmbda*MM
    b=np.matmul(lmbda*MM, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )) - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) ) 
    #print(b[:10])'''
    A = sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  )) - lmbda*MM
    b=sparse.csr_matrix.dot(lmbda*MM, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )) 
    - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) )
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
    #print(npixels)
    #FU=np.zeros((npixels,npixels),dtype=np.float32)
    #FV=np.zeros((npixels,npixels),dtype=np.float32)
    FU=sparse.csr_matrix((npixels,npixels),dtype=np.float32)
    FV=sparse.csr_matrix((npixels,npixels),dtype=np.float32)
    quadr_ov_x=np.vectorize(deriv_quadra_over_x)
    for i in range(len(S)):
        M=conv_matrix(S[i],sz)
        #print(M)
        #u_=np.matmul(M,np.reshape((u+du),(npixels,1),'F'))
        #v_=np.matmul(M,np.reshape((v+dv),(npixels,1),'F'))
        u_=sparse.csr_matrix.dot(M,np.reshape((u+du),(npixels,1),'F'))
        v_=sparse.csr_matrix.dot(M,np.reshape((v+dv),(npixels,1),'F'))

        pp_su=quadr_ov_x(u_,1)
        pp_sv=quadr_ov_x(v_,1)
        #print(pp_sv)
        
        FU        = FU+ sparse.csr_matrix.dot(M.T,sparse.csr_matrix.dot(spdiags(pp_su.T, 0, npixels, npixels),M));
        FV        = FV+ sparse.csr_matrix.dot(M.T,sparse.csr_matrix.dot(spdiags(pp_sv.T, 0, npixels, npixels),M));
    
    MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.csr_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.csr_matrix((npixels,npixels)) , -FV ) )  ))  
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
    duu = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Iy2,(npixels,1),'F')
    
    dvv = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Ixy,(npixels,1),'F')
    
    dduv = spdiags(tmp.T, 0, npixels, npixels)

    A = sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  )) - lmbda*MM
    b=sparse.csr_matrix.dot(lmbda*MM, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )) 
    - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) ) 

    if( ((np.max(pp_su)-np.max(pp_su)<1E-06)   ) and ((np.max(pp_sv)-np.max(pp_sv)<1E-06)   ) and ((np.max(pp_d)-np.max(pp_d)<1E-06)   ) ):
        iterative=False
    else:
        iterative=True
    return [A,b,iterative]
#############################################################
def  compute_flow_base(Image1,Image2,max_iter,max_linear_iter,u,v,alpha,lmbda,S,size_median_filter,h,coef):
    npixels=u.shape[0]*u.shape[1]
    
    for i in range(max_iter):
        du=np.zeros((u.shape)); dv=np.zeros((v.shape))
        [Ix,Iy,It]=derivatives(Image1,Image2,u,v,h,coef)
        print('deriv')
        print(It)
        print(Ix)
        print(Iy)
        #print('iter',i,'shapes',Ix.shape)
        for j in range(max_linear_iter):
            if (alpha==1):
                [A,b,iterative]=flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda)
            elif((alpha>0)  and (alpha !=1) ):
                [A,b,iterative]=flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda)

                [A1,b1,iterative1]=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda)
                
                A=alpha*A+(1-alpha)*A1
                b=alpha*b+(1-alpha)*b1
            elif(alpha==0):
                [A,b,iterative]=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda)
            
            #x=np.matmul(np.linalg.inv(A),b)
            #x=np.linalg.solve(A,b)
            x=scipy.sparse.linalg.spsolve(A,b)
            du=np.reshape(x[0:npixels], (u.shape[0],u.shape[1]),'F' )
            dv=np.reshape(x[npixels:2*npixels], (u.shape[0],u.shape[1]),'F' )
            
            u0=u; v0=v
            u=u+du;v=v+dv
            #print('u avant')
            #print(u)
            #u=medfilt(u,size_median_filter)
            u=median_filter(u,size=(5,5))
            #print('u après')
            #print(u)
            #v=medfilt(v,size_median_filter)
            v=median_filter(v,size=(5,5))
            du = u- u0
            dv = v- v0
            '''print('du')
            print(du)
            print('dv')
            print(dv)'''
            u= u0
            v=v0
            
        print('it',i)
        u = u + du
        v = v + dv
        '''print('u')
        print(u)
        print('v')
        print(v)'''

        
    return [u,v]


#############################################################
S=[];
S.append(np.array([[1,-1]]))
S.append(np.array([[1],[-1]]))

'''It =np.array([[-3.69547,   0.41807  , 2.02836 , -0.53392 ,  0.00000],
  [ 1.63937 , -1.49628 , -2.13478  , 0.14007 ,  0.00000],
  [-0.99281 , -0.62484 , -1.55485 ,  0.38739 ,  0.00000],
  [-0.19625 , -1.58552 ,  0.51390 , -0.05108 ,  0.00000],
  [ 0.00000,   0.00000 ,  0.00000 ,  0.00000,   0.00000]])'''

'''Image1 = np.array([[ 1.66114 ,  1.46993 , -0.59511  , 0.65337 ,  0.00000],
  [-1.06698  ,-0.46049 ,  0.94558  , 0.91834  , 0.00000],
   [0.86646,   0.20117  ,-0.89750  , 0.28706 ,  0.00000],
  [-0.21850  , 0.66434  , 1.01487  ,-0.38705  , 0.00000],
  [ 0.00000 ,  0.00000  , 0.00000 ,  0.00000  , 0.00000]])

Image2= np.array([[ 1.50229  ,-1.37956  ,-0.67601 ,  0.18091  , 0.00000],
  [ 1.21001 ,  0.56375 , -0.12846 , -0.15330   ,0.00000],
  [-0.80391 ,  0.31335 ,  0.45995 , -0.00943  , 0.00000],
 [ -0.18017 , -1.22093  , 0.84927  , 2.06008   ,0.00000],
  [ 0.00000  , 0.00000  , 0.00000 ,  0.00000  , 0.00000]])'''
Image1=np.array([[6.74449552, 5.53346511, 8.68140734, 2.04012167, 1.7470887 ],
       [1.35848093, 4.86814148, 8.45172817, 2.33243605, 5.19957588],
       [7.40448257, 0.43829789, 9.42031693, 9.64404467, 9.46878689],
       [7.40480588, 0.9560134 , 8.75912932, 2.98090913, 8.97988959],
       [9.80286916, 4.68173038, 7.37135071, 9.13694127, 8.421006  ]])

Image2=np.array([[7.6243546 , 8.76405184, 0.93504855, 9.80596673, 8.81201369],
       [5.77520562, 7.52720659, 7.20116081, 0.80966906, 0.56304753],
       [2.86931799, 4.53189126, 9.24839657, 3.15723941, 2.6250291 ],
       [1.06423688, 7.17828962, 8.30585961, 1.51904165, 0.60133615],
       [4.94036009, 8.48463488, 9.34919316, 4.0668727 , 9.53326332]])



u=np.zeros((Image1.shape))
v=np.zeros((Image1.shape))
'''du=np.zeros((u.shape))
dv=np.zeros((u.shape))'''

lmbda=1
h=np.array([[-1 ,8, 0 ,-8 ,1 ]])
h=1/12*h
coef=0.5

u,v=compute_flow_base(Image1,Image2,10,1,u,v,0.5,lmbda,S,5,h,coef)
print(u)
print(v)
u=np.array(u,dtype=np.float32)
v=np.array(v,dtype=np.float32)
I=cv2.remap(Image1,u,v,cv2.INTER_CUBIC)
print('norm: ',np.linalg.norm(I-Image2)/np.linalg.norm(Image2))
print('I')
print(I)
print('Image2')
print(Image2)