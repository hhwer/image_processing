
import pdb
import numpy as np
import scipy
from scipy.ndimage import convolve




def mysolver(u_hat,C,K=1,mode_L=1,T=1):

    max_step = int(T//C)
    u = u_hat.copy()
    for i in range(max_step):
#        u = ite_heat(u,C)
        print('T=%f'%(C*i))
#        u = ite_PM(u, C, K=K)
        u = ite_filter(u,C,mode_L=mode_L)
    u[u>1.0]=1.0
    u[u<0.0]=0
    return u


def ite_heat(u,C):
    b = np.zeros([u.shape[0]+2,u.shape[1]+2,u.shape[2]])
    b[1:-1,1:-1,:] = u.copy()
    b[0,:,:]=b[1,:,:].copy()
    b[-1,:,:]=b[-2,:,:].copy()
    b[:,0,:]=b[:,1,:].copy()
    b[:,-1,:]=b[:,-2,:].copy()
    
    u = u + C*(b[:-2,1:-1,:]+b[2:,1:-1,:]+b[1:-1,:-2,:]+b[1:-1,2:,:]-4*u)
    return(u)

def ite_PM(u, C, K=1):
    u_temp = np.zeros([u.shape[0]+2,u.shape[1]+2,u.shape[2]])
    u_temp[1:-1,1:-1,:] = u.copy()
    u_temp[0,:,:]=u_temp[1,:,:].copy()
    u_temp[-1,:,:]=u_temp[-2,:,:].copy()
    u_temp[:,0,:]=u_temp[:,1,:].copy()
    u_temp[:,-1,:]=u_temp[:,-2,:].copy()
    
    u1 = u_temp[1:-1,2:,:]-u_temp[1:-1,1:-1,:]
    u2 = u_temp[2:,1:-1,:]-u_temp[1:-1,1:-1]
    s = u1*u1+u2*u2
#    print(np.max(s))
    c = 1/(1.0+s/K)
    b = np.zeros([u.shape[0]+2,u.shape[1]+2,u.shape[2]])
    b[1:-1,1:-1,:] = c.copy()
    b[0,:,:]=b[1,:,:].copy()
    b[-1,:,:]=b[-2,:,:].copy()
    b[:,0,:]=b[:,1,:].copy()
    b[:,-1,:]=b[:,-2,:].copy()
    div_b_grad_u = b[2:,1:-1,:]*u_temp[2:,1:-1,:]\
        + b[1:-1,1:-1,:]*(u_temp[:-2,1:-1,:]+u_temp[1:-1,:-2,:])\
        + b[1:-1,2:,:]*u_temp[1:-1,2:,:]\
        - (b[2:,1:-1,:]+b[1:-1,2:,:]+2*b[1:-1,1:-1,:])*u_temp[1:-1,1:-1,:]
    u = u + C*div_b_grad_u
    return u

def ite_filter(u,C,mode_L=1):
    u_temp = np.zeros([u.shape[0]+2,u.shape[1]+2,u.shape[2]])
    u_temp[1:-1,1:-1,:] = u.copy()
    u_temp[0,:,:]=u_temp[1,:,:].copy()
    u_temp[-1,:,:]=u_temp[-2,:,:].copy()
    u_temp[:,0,:]=u_temp[:,1,:].copy()
    u_temp[:,-1,:]=u_temp[:,-2,:].copy()
    
    u11 = u_temp[1:-1,2:,:]-u_temp[1:-1,1:-1,:]
    u12 = u_temp[1:-1,1:-1,:]-u_temp[1:-1,:-2,:]
    u21 = u_temp[2:,1:-1,:]-u_temp[1:-1,1:-1]
    u22 = u_temp[1:-1,1:-1,:]-u_temp[:-2,1:-1]
    m1 = minmod(u11,u12)
    m2 = minmod(u21,u22)
    
    s = m1*m1+m2*m2
    
    if mode_L==1:
        L = u_temp[:-2,1:-1,:]+u_temp[2:,1:-1,:]+u_temp[1:-1,:-2,:]+u_temp[1:-1,2:,:]-4*u
    else:
        uxx = u_temp[:-2,1:-1,:]+u_temp[2:,1:-1,:]-2*u
        uyy = u_temp[1:-1,:-2,:]+u_temp[1:-1,2:,:]-2*u
        m = np.zeros([u.shape[0],u.shape[1]+1,u.shape[2]])
        m[:,:-1,:] = m2.copy()
        m[:,-1,:] = m[:,-2,:].copy()
        uxy = m[:,1:,:]-m[:,:-1,:]
        L = (m1*m1*uxx + 2*m1*m2*uxy + m2*m2*uyy)/np.sqrt(s*s+1e-10)    

    s = np.sqrt(s)
    print(np.max(s))
    u = u - C*s*L
    return u

def minmod(u1,u2):
    return (np.sign(u1*u2)+1)//2*np.sign(u1)*np.minimum(abs(u1),abs(u2))

def fspecial(kernel_size,gaussian_sigma):
    K_g = np.zeros([kernel_size,kernel_size])
    i_a = np.zeros([kernel_size,1])
    j_a = np.zeros([1,kernel_size])
    sigma = gaussian_sigma**2
    if sigma == 0:
        kernel = K_g.copy()
        mid = (kernel_size-1)//2
        kernel[mid,mid] = 1
        return kernel
    for i in range(kernel_size):
        i_a[i,0] = (i-(kernel_size-1)/2)**2 
        j_a[0,i] = (i-(kernel_size-1)/2)**2
    K_g = np.exp((-i_a-j_a)/sigma)
    sig = np.sum(K_g)
    kernel = K_g / sig
#    pdb.set_trace()
    return kernel

def conv(f,kernel,mode='constant'):
    pixel_size = f.shape[2]
    b = np.zeros(f.shape)
    for i in range(pixel_size):
        b[:,:,i] = convolve(f[:,:,i],kernel,mode=mode)
    return b

def add_NoiseAndBluf(f,kernel,mode='constant',noise_rate=100):
    u_hat = conv(f,kernel,mode) + np.max(f)* noise_rate * np.random.randn(*f.shape)
    u_hat[u_hat>1.] = 1.
    u_hat[u_hat<0.] = 0.
    return u_hat

def laplace(u,mode='constant'):
    b = np.zeros([u.shape[0]+2,u.shape[1]+2,u.shape[2]])
    b[1:-1,1:-1,:] = u.copy()
    if (mode=='reflect' or mode == 'nearest'):
        b[0,:,:]=b[1,:,:].copy()
        b[-1,:,:]=b[-2,:,:].copy()
        b[:,0,:]=b[:,1,:].copy()
        b[:,-1,:]=b[:,-2,:].copy()
    
    u1 = b[:-2,1:-1,:]+b[2:,1:-1,:]+b[1:-1,:-2,:]+b[1:-1,2:,:]-4*u 
    return u1

def grad(u,mode='constant'):
    b = np.zeros([u.shape[0]+1,u.shape[1]+1,u.shape[2]])
    b[:-1,:-1,:] = u.copy()
    if (mode=='reflect' or mode == 'nearest'):
        b[-1,:,:]=b[-2,:,:].copy()
        b[:,-1,:]=b[:,-2,:].copy()
    
    u1 = b[:-1,1:,:]-b[:-1,:-1,:]
    u2 = b[1:,:-1,:]-b[:-1,:-1]
    return u1,u2

def div(u1,u2,mode='constant'):
    b1 = np.zeros([u1.shape[0],u1.shape[1]+1,u1.shape[2]])
    b2 = np.zeros([u1.shape[0]+1,u1.shape[1],u1.shape[2]])
    b1[:,1:,:] = u1.copy()
    b2[1:,:,:] = u2.copy()

    if (mode=='reflect' or mode == 'nearest'):
        b1[:,0,:]=b1[:,1,:].copy()
        b2[0,:,:]=b2[1,:,:].copy()
    
    u = b1[:,1:,:] - b1[:,:-1,:] + b2[1:,:,:]-b2[:-1,:,:]
    return u
    
