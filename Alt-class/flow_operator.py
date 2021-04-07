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
    '''print('warped')
    print(WImage[0:10,0:10])'''
    Ix=filter2(Image, h)
    Iy=filter2(Image, h.T)
    
    Iy=cv2.remap(Iy,XI,YI,interpolation=cv2.INTER_CUBIC)   
    Ix=cv2.remap(Ix,XI,YI,interpolation=cv2.INTER_CUBIC)
    '''print('I2x')
    print(Ix[0:10,0:10])
    print('I2y')
    print(Iy[0:10,0:10])'''
    return [WImage,Ix,Iy]
    
############################################
def derivatives(Image1,Image2,u,v,h,b):
    N,M=Image1.shape
    y=np.linspace(0,N-1,N)
    x=np.linspace(0,M-1,M)
    x,y=np.meshgrid(x,y)
    Ix=np.zeros((N,M))
    Iy=np.zeros((N,M))
    x=x+u; y=y+v
    WImage,I2x,I2y=warp_image2(Image2,x,y,h)  # Derivatives of the secnd image 

    It= WImage-Image1 # Temporal deriv
    
    Ix=filter2(Image1, h) # spatial derivatives for the first image 
    Iy=filter2(Image1, h.T)
    '''print('I1x')
    print(Ix[0:10,0:10])
    print('I1y')
    print(Iy[0:10,0:10])'''
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
    M=sparse.lil_matrix( ( sz[0]*sz[1],sz[0]*sz[1] ),dtype=np.float32)
    if( F.shape==(1,2)):
        for i in range(sz[0],sz[0]*sz[1]):      
            M[i,i-sz[0]]=-1     
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
def deriv_charbonnier_over_x(x,sigma,a):
     y =2*a*(sigma**2 + x**2)**(a-1);
     return y
def deriv_quadra_over_x(x,sigma):
    y = 2 / (sigma**2)
    return y
########################################################
def flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda):
    ''' Using Charbonnier function '''
    sz=np.shape(Ix)
    npixels=Ix.shape[1]*Ix.shape[0]
    
    FU=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    FV=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    for i in range(len(S)):
        M=conv_matrix(S[i],sz)
        u_=sparse.lil_matrix.dot(M,np.reshape((u+du),(npixels,1),'F'))
        v_=sparse.lil_matrix.dot(M,np.reshape((v+dv),(npixels,1),'F'))
        pp_su=deriv_charbonnier_over_x(u_,0.001,0.5)
        pp_sv=deriv_charbonnier_over_x(v_,0.001,0.5)
        
        FU        = FU+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_su.T, 0, npixels, npixels),M))
        FV        = FV+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_sv.T, 0, npixels, npixels),M))
    MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FV ) )  ))  

    Ix2 = Ix*Ix
    Iy2 = Iy*Iy
    Ixy = Ix*Iy
    Itx = It*Ix
    Ity = It*Iy

    It = It + Ix*du+ Iy*dv
    
    #pp_d=deriv_lorentz_over_x(np.reshape(It,(npixels,1),'F'),1.5)
    pp_d=deriv_charbonnier_over_x(np.reshape(It,(npixels,1),'F'),0.001,1.5)
    
    tmp=pp_d*np.reshape(Ix2,(npixels,1),'F')
    duu = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Iy2,(npixels,1),'F')
    
    dvv = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Ixy,(npixels,1),'F')
    
    dduv = spdiags(tmp.T, 0, npixels, npixels)

    A = sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  )) - lmbda*MM
    b=sparse.lil_matrix.dot(lmbda*MM, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )) - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) )
    if( ((np.max(pp_su)-np.max(pp_su)<1E-06)   ) and ((np.max(pp_sv)-np.max(pp_sv)<1E-06)   ) and ((np.max(pp_d)-np.max(pp_d)<1E-06)   ) ):
        iterative=False
    else:
        iterative=True
    return [A,b,iterative]
#############################################################
def flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda):
    '''using quadratic function ''' 
    sz=np.shape(Ix)
    npixels=Ix.shape[1]*Ix.shape[0]

    FU=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    FV=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    quadr_ov_x=np.vectorize(deriv_quadra_over_x)
    for i in range(len(S)):
        M=conv_matrix(S[i],sz)
        
        u_=sparse.lil_matrix.dot(M,np.reshape((u+du),(npixels,1),'F'))
        v_=sparse.lil_matrix.dot(M,np.reshape((v+dv),(npixels,1),'F'))

        pp_su=quadr_ov_x(u_,1)
        pp_sv=quadr_ov_x(v_,1)
        
        FU        = FU+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_su.T, 0, npixels, npixels),M))
        FV        = FV+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_sv.T, 0, npixels, npixels),M))
    
    MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FV ) )  ))  
   
    Ix2 = Ix*Ix
    Iy2 = Iy*Iy
    Ixy = Ix*Iy
    Itx = It*Ix
    Ity = It*Iy

    It = It + Ix*du+ Iy*dv
    
    pp_d=deriv_quadra_over_x(np.reshape(It,(npixels,1),'F'),(1.5/0.03))
    
    tmp=pp_d*np.reshape(Ix2,(npixels,1),'F')
    duu = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Iy2,(npixels,1),'F')
    
    dvv = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Ixy,(npixels,1),'F')
    
    dduv = spdiags(tmp.T, 0, npixels, npixels)

    #A = sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  )) - lmbda*MM
    A = sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  ))
    print('AA')
    print(A)
    print('MM')
    print(MM)
    A=A-lmbda*MM
    
    b=sparse.lil_matrix.dot(lmbda*MM, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )) - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) ) 

    if( ((np.max(pp_su)-np.max(pp_su)<1E-06)   ) and ((np.max(pp_sv)-np.max(pp_sv)<1E-06)   ) and ((np.max(pp_d)-np.max(pp_d)<1E-06)   ) ):
        iterative=False
    else:
        iterative=True
    return [A,b,iterative]
#############################################################
def  compute_flow_base(Image1,Image2,max_iter,max_linear_iter,u,v,alpha,lmbda,S,size_median_filter,h,coef,uhat,vhat):
    npixels=u.shape[0]*u.shape[1]
    u0=np.zeros((u.shape)); v0=np.zeros((u.shape));
    for i in range(max_iter):
        du=np.zeros((u.shape)); dv=np.zeros((v.shape))
       
        [Ix,Iy,It]=derivatives(Image1,Image2,u,v,h,coef)
        print('deriv')
        #print(It)
        print('It')
        print(It[0:5,0:5])
        print('Ix')
        print(Ix[0:5,0:5])
        print('Iy')
        print(Iy[0:5,0:5])
        #print('iter',i,'shapes',Ix.shape)
        for j in range(max_linear_iter):
            if (alpha==1):
                [A,b,iterative]=flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda)
            elif(alpha>0 and alpha != 1):
                [A,b,iterative]=flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda)

                [A1,b1,iterative1]=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda)
                
                A=alpha*A+(1-alpha)*A1
                b=alpha*b+(1-alpha)*b1
            elif(alpha==0):
                [A,b,iterative]=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda)
            print('A')
            print(A)

            '''tmpu=deriv_charbonnier_over_x(u-uhat,0.001,0.5)
            tmpv=deriv_charbonnier_over_x(v-vhat,0.001,0.5)
            tmpA=spdiags(np.reshape(np.hstack((tmpu,tmpv)),(1,A.shape[0]*A.shape[1]),'F'),0,A.shape[0],A.shape[1])'''
            tmp0=np.reshape( np.hstack((u-uhat,v-vhat)) , (1,A.shape[0]*A.shape[1]) ,'F')
            tmp1=deriv_charbonnier_over_x(tmp0,0.001,0.5)
            tmpA=spdiags(tmp1,0,A.shape[0],A.shape[1])
            A= A + lambda2*tmpA
            b=b - lambda2*tmp1*tmp0;    


            #x=np.matmul(np.linalg.inv(A),b)
            #x=np.linalg.solve(A,b)
            x=scipy.sparse.linalg.spsolve(A,b)
            print('x')
            print(x[:5])
            x[x>1]=1
            x[x<-1]=-1
            du=np.reshape(x[0:npixels], (u.shape[0],u.shape[1]),'F' )
            dv=np.reshape(x[npixels:2*npixels], (u.shape[0],u.shape[1]),'F' )
            print('dv')
            print(dv[:5,:5])
            u0=u; v0=v
            u=u+du;v=v+dv
            #print('u avant')
            #print(u)
            #u=medfilt(u,size_median_filter)
            #print('u après')
            #print(u)
            #v=medfilt(v,size_median_filter)
            if (size_median_filter !=0):
                u=median_filter(u,size=(size_median_filter,size_median_filter))
                v=median_filter(v,size=(size_median_filter,size_median_filter))
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
'''S=[]
S.append(np.array([[1,-1]]))
S.append(np.array([[1],[-1]]))'''

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
'''Image1=np.array([[6.74449552, 5.53346511, 8.68140734, 2.04012167, 1.7470887 ],
       [1.35848093, 4.86814148, 8.45172817, 2.33243605, 5.19957588],
       [7.40448257, 0.43829789, 9.42031693, 9.64404467, 9.46878689],
       [7.40480588, 0.9560134 , 8.75912932, 2.98090913, 8.97988959],
       [9.80286916, 4.68173038, 7.37135071, 9.13694127, 8.421006  ]])

Image2=np.array([[7.6243546 , 8.76405184, 0.93504855, 9.80596673, 8.81201369],
       [5.77520562, 7.52720659, 7.20116081, 0.80966906, 0.56304753],
       [2.86931799, 4.53189126, 9.24839657, 3.15723941, 2.6250291 ],
       [1.06423688, 7.17828962, 8.30585961, 1.51904165, 0.60133615],
       [4.94036009, 8.48463488, 9.34919316, 4.0668727 , 9.53326332]])'''


#Image1=cv2.imread(r'/home/achabib/Bureau/Secrets_of/Lorentzian/SPARSE/image1.png',0)
#Image2=cv2.imread(r'/home/achabib/Bureau/Secrets_of/Lorentzian/SPARSE/image2.png',0)
''''Image1=cv2.imread(r'/home/achabib/Bureau/Secrets_of/Lorentzian/SPARSE/Im1.png',0)
Image2=cv2.imread(r'/home/achabib/Bureau/Secrets_of/Lorentzian/SPARSE/Im2.png',0)
Image1=np.array(Image1,dtype=np.float32)
Image2=np.array(Image2,dtype=np.float32)

u=np.zeros((Image1.shape))
v=np.zeros((Image1.shape))'''
'''du=np.zeros((u.shape))
dv=np.zeros((u.shape))'''

'''lmbda=1
h=np.array([[-1 ,8, 0 ,-8 ,1 ]])
#[1 -8 0 8 -1]/12

h=h/12
coef=0.5
u,v=compute_flow_base(Image1,Image2,10,1,u,v,0.5,lmbda,S,5,h,coef)
print('u')
print(u[0:10,0:10])
print('v')
print(v[0:10,0:10])
u=np.array(u,dtype=np.float32)
v=np.array(v,dtype=np.float32)

N,M=Image1.shape
y=np.linspace(0,N-1,N)
x=np.linspace(0,M-1,M)
x,y=np.meshgrid(x,y)
x=x+u; y=y+v
x=np.array(x,dtype=np.float32)
y=np.array(y,dtype=np.float32)
I=cv2.remap(np.array(Image1,dtype=np.float32),x,y,cv2.INTER_CUBIC)
print('norm: ',np.linalg.norm(I-Image2)/np.linalg.norm(Image2))
print('I')
print(I[0:10,0:10])
print('Image2')
print(Image2[0:10,0:10])
cv2.imwrite('warped.png',I)'''