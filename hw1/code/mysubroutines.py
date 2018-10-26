
import pdb
import numpy as np
import scipy
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from scipy.sparse import linalg
from scipy.sparse.linalg import LinearOperator

def mysolver(u_hat,kernel,lambda_weight,mode='constant'):
    mu = 1e-8
    tol = 1e-3
#    delta = (5**(1/2)+1)/2
    delta = 0.5
    function = A(mu,u_hat.shape,kernel,mode)
    m,n,l = u_hat.shape
    function_A = LinearOperator((m*n*l,m*n*l),matvec=function)
    ATf = np.reshape(conv(u_hat,kernel,mode=mode),-1)
    u = u_hat.copy()
    d1,d2 = grad(u,mode)
    b1 = np.zeros(u.shape)
    b2 = np.zeros(u.shape)
    tol = tol**2 * np.sum(u_hat*u_hat)
    for i in range(20):
        ite = 3
        cg_tol = 0.5*1e-3
        u,d1,d2,b1,b2,Wu1,Wu2 = Admm(                                                  function_A,u,d1,d2,b1,b2,ATf,lambda_weight,mu,mode=mode,delta=delta,ite=ite,cg_tol=cg_tol)
        err = np.sum((Wu1-d1)*(Wu1-d1))+np.sum((Wu2-d2)*(Wu2-d2))
        print(err,i)
        if err < tol:
            break
#    pdb.set_trace()
    return u


def fspecial(kernel_size,gaussian_sigma):
    K_g = np.zeros([kernel_size,kernel_size])
    i_a = np.zeros([kernel_size,1])
    j_a = np.zeros([1,kernel_size])
    sigma = gaussian_sigma**2
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
#        b[:,:,i] = convolve2d(f[:,:,i],kernel,boundary = 'symm',mode='same')
    return b

def add_NoiseAndBluf(f,kernel,lambda_weight,mode='constant',noise_rate=100):
    u_hat = conv(f,kernel,mode) + np.max(f)/noise_rate * np.random.randn(*f.shape)
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

def Tau(tau, nu):
    return np.sign(nu) * np.maximum(abs(nu)-tau,0)

def A(mu,u_shape,kernel,mode='constant'):
    def f(x):
        x = x.reshape(u_shape)
        return np.reshape(conv(conv(x,kernel,mode=mode),kernel,mode=mode) + mu*laplace(x,mode=mode),-1)
##        return np.reshape(conv(conv(x,kernel,mode=mode),kernel,mode=mode) + mu*div(*grad(x,mode),mode),-1)
    return f

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
    
def Admm(function_A,u,d1,d2,b1,b2,ATf,lamb,mu,mode='constant',delta=1,ite=8,cg_tol=1e-3):
    b0 = ATf + np.reshape(mu*div(d1-b1,d2-b2,mode=mode),-1)
    u = u.reshape(-1)
    u,infos = linalg.cgs(function_A,b0,x0=u,maxiter=ite,tol=cg_tol)
#    u,infos = linalg.cgs(function_A,b0,maxiter=ite,tol=cg_tol)
#    print(infos)
#    print(np.linalg.norm(function_A(u)-b0)/np.linalg.norm(b0))
    u = u.reshape(d1.shape)
#    u = (u-np.min(u)) / (np.max(u)-np.min(u))
    u = u/np.max(abs(u))
    u[u>1.0]=1.0
    u[u<0.0]=0
    Wu1,Wu2 = grad(u,mode=mode)
    d1 = Tau(lamb/mu, Wu1+b1)
    d2 = Tau(lamb/mu, Wu2+b2)
    b1 = b1 + delta*(Wu1-d1)
    b2 = b2 + delta*(Wu2-d2)
    return u,d1,d2,b1,b2,Wu1,Wu2

