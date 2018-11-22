
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import pdb
from scipy import misc
import mysubroutines
argc = len(sys.argv)


K = 100
C = 5e-2
image_path = (sys.argv[1]) if argc>1 else 'image/fig1.png'
T = float(sys.argv[2]) if argc>2 else 1
result_image_path=(sys.argv[3]) if argc>3 else 'image/fig1_result.png'
mode_L=int(sys.argv[4]) if argc>4 else 1
raw_isnoise = int(sys.argv[5]) if argc>5 else 0
kernel_size= 15
gaussian_sigma=1.5
noise_rate = 0.01
kernel=mysubroutines.fspecial(kernel_size,gaussian_sigma)


f=mpimg.imread(image_path)



if len(f.shape) < 3:
    f = np.reshape(f,[*f.shape,1])
if np.max(f)>1.0:
    f = f/255

if raw_isnoise == 0:
    u_hat = mysubroutines.add_NoiseAndBluf(
                             f,kernel,noise_rate=noise_rate)
else:
    u_hat = f


u1 = u_hat.copy()
if f.shape[2] == 1:
    u1 = u_hat[:,:,0]
misc.imsave('image/fig1_noise.png', u1)
plt.close('all')
h1 = plt.figure()
ax1 = h1.add_subplot(121)
ax1.imshow(u1, cmap= plt.cm.gray)



u=mysubroutines.mysolver(u_hat,C=C,K=K,mode_L=mode_L,T=T)
u2 = u.copy()
if f.shape[2] == 1:
    u2 = u2[:,:,0]
misc.imsave(result_image_path,u2)

