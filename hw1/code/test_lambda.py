
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mysubroutines
argc = len(sys.argv)

image_path =  'image/fig2.png'
lambda_weight= float(sys.argv[1]) if argc>1 else 1e+2
kernel_size= 15
gaussian_sigma=float(sys.argv[4]) if argc>4 else 1.5
result_image_path=(sys.argv[5]) if argc>5 else 'image/fig2_result.png'
mode=(sys.argv[6]) if argc>6 else 'constant'
noise_rate = float(sys.argv[7]) if argc>7 else 100
kernel=mysubroutines.fspecial(kernel_size,gaussian_sigma)
f=mpimg.imread(image_path)
if len(f.shape) < 3:
    f = np.reshape(f,[*f.shape,1])
if np.max(f)>1.0:
    f = f/255



max_step= int(sys.argv[2]) if argc>2 else 10

u_hat = mysubroutines.add_NoiseAndBluf(
                             f,kernel,lambda_weight,mode=mode,noise_rate=noise_rate)


h=plt.figure()
ax=h.add_subplot(111)
u1 = u_hat.copy()
if f.shape[2] == 1:
    u1 = u_hat[:,:,0]

ax.imshow(u1,cmap=plt.cm.gray)
h.savefig('image/fig2_noise.png')


u=mysubroutines.mysolver(u_hat,kernel,lambda_weight,mode=mode,max_step=max_step)
h=plt.figure()
ax=h.add_subplot(111)
u1 = u.copy()
if f.shape[2] == 1:
    u1 = u[:,:,0]
ax.imshow(u1,cmap=plt.cm.gray)
#ax.imshow(u)
h.savefig('image/fig2_result_lambda'+str(lambda_weight)+'_ite'+str(max_step)+'.png')

