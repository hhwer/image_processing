

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import mysubroutines
argc = len(sys.argv)

image_path = (sys.argv[1]) if argc > 1 else 'image/fig3.png'
T = float(sys.argv[2]) if argc > 2 else 1
result_image_path = (sys.argv[3]) if argc > 3 else 'image/fig3_result.png'

f = mpimg.imread(image_path)


if np.max(f) > 1.0:
    f = f/255
u_hat = 0.2989*f[:, :, 0] + 0.5870*f[:, :, 1] + 0.1140*f[:, :, 2]



alpha = 1.1
freq_reinitial = 2
step_reinitial = 20
max_step = 500
# u = mysubroutines.level_set1(u1)
# u = mysubroutines.level_set_central(u1, alpha, freq_reinitial, step_reinitial, max_step)
# u = mysubroutines.level_set_central_h1(u1, alpha, freq_reinitial, step_reinitial, max_step)
# misc.imsave(result_image_path, u)
# plt.figure()
# plt.contour(u[::-1, :], [0])
# plt.show()

mu = 0.9
_lambda = 5e-1


u = mysubroutines.cv(u_hat, mu, _lambda, alpha=10, eps=1e-6, max_step=max_step)


plt.contour(u[::-1, :], [mu])
plt.figure()
plt.imshow(u)
plt.show()

