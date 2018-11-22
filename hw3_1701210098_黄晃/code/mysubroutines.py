
import matplotlib.pyplot as plt
import numpy as np


def level_set_central_h1(u0, alpha, freq_reinitial, step_reinitial, max_steps):
    # |grad u0|^2
    h = 1
    cfl = 1.0/2
    delta_t = cfl*h
    b = np.zeros([u0.shape[0]+2, u0.shape[1]+2])
    b[1:-1, 1:-1] = u0.copy()
    b[0, :] = b[1, :].copy()
    b[-1, :] = b[-2, :].copy()
    b[:, 0] = b[:, 1].copy()
    b[:, -1] = b[:, -2].copy()
    u0x = (b[1:-1, 2:]-b[1:-1, :-2])/2
    u0y = (b[2:, 1:-1]-b[:-2, 1:-1])/2
    square_grad_u0 = np.square(u0x)+np.square(u0y)
    g = 1.0/(1+square_grad_u0*1e+3)   # g(|grad_u0|^2)
    # grad g
    b = np.zeros([u0.shape[0]+2, u0.shape[1]+2])
    b[1:-1, 1:-1] = g.copy()
    b[0, :] = b[1, :].copy()
    b[-1, :] = b[-2, :].copy()
    b[:, 0] = b[:, 1].copy()
    b[:, -1] = b[:, -2].copy()
    gx = (b[1:-1, 2:]-b[1:-1, :-2])/2
    gy = (b[2:, 1:-1]-b[:-2, 1:-1])/2
    max_g0 = np.maximum(g, 0)
    min_g0 = np.minimum(g, 0)
    max_gx0 = np.maximum(gx, 0)
    min_gx0 = np.minimum(gx, 0)
    max_gy0 = np.maximum(gy, 0)
    min_gy0 = np.minimum(gy, 0)
    u = np.ones(u0.shape)*(-10)
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    for i in range(max_steps):
        if i % freq_reinitial == 0:
            u = reinitial_2d(u, step_reinitial)
        if i % 10 == 0:
            plt.cla()
            plt.imshow(u0, cmap=plt.cm.gray)
            plt.contour(u, [0],  colors='r')
            plt.pause(0.001)

        b = np.zeros([u0.shape[0]+2, u0.shape[1]+2])
        b[1:-1, 1:-1] = u.copy()
        b[0, :] = b[1, :].copy()
        b[-1, :] = b[-2, :].copy()
        b[:, 0] = b[:, 1].copy()
        b[:, -1] = b[:, -2].copy()
        print(np.max(b))
        yp = (b[2:, 1:-1] - u)
        yn = (u - b[:-2, 1:-1])
        xp = (b[1:-1, 2:] - u)
        xn = (u - b[1:-1, :-2])
        godnov_p = np.square(np.maximum(xn, 0)) + np.square(np.maximum(yn, 0))
        godnov_p += np.square(np.minimum(xp, 0)) + np.square(np.minimum(yp, 0))
        godnov_p = np.sqrt(godnov_p)
        godnov_n = np.square(np.minimum(xn, 0)) + np.square(np.minimum(yn, 0))
        godnov_n += np.square(np.maximum(xp, 0)) + np.square(np.maximum(yp, 0))
        godnov_n = np.sqrt(godnov_n)
        k_curve = k_curve_h1(b)
        u = u + delta_t*(g*k_curve
                         + alpha*(max_g0*godnov_p+min_g0*godnov_n)
                         + max_gx0*xn+min_gx0*xp+max_gy0*yn+min_gy0*yp)
    return u


def cv(f, mu, _lambda, alpha=0, eps=1e-6, max_step=1000):
    u0 = np.zeros(f.shape)
    d1 = np.zeros(f.shape)
    d2 = np.zeros(f.shape)
    b1 = np.zeros(f.shape)
    b2 = np.zeros(f.shape)
    grad_f1, grad_f2 = grad(f)
    g = 1/(1+alpha*(np.square(grad_f1)+np.square(grad_f2)))
    for k in range(max_step):
        u = u0.copy()
        r = get_r(f, u0, mu)
        for i in range(1):
            u = gs(mu, _lambda, r, u, d1, d2, b1, b2)
        error = np.linalg.norm(u-u0)
        u0 = u.copy()
        print(error)
        if error < eps and k > 10:
            break
        grad_u1, grad_u2 = grad(u)
        d1, d2 = shrink(grad_u1+b1, grad_u2+b2, _lambda, g)
#        d1 = shrink_alone(grad_u1+b1, _lambda, g)
#        d2 = shrink_alone(grad_u2+b2, _lambda, g)
        b1 = b1 + grad_u1 - d1
        b2 = b2 + grad_u2 - d2
    return u


def get_r(f, u, mu):
    u_ge_mu = u > mu
    u_le_mu = 1 - u_ge_mu
    c1 = np.sum(u_ge_mu*f) / (np.sum(u_ge_mu)+1e-10)
    c2 = np.sum(u_le_mu*f) / (np.sum(u_le_mu)+1e-10)
#    c1 = np.sum(f*u)/(np.sum(u)+1e-10)
#    c2 = np.sum(f*(1-u))/(np.sum(1-u)+1e-10)
    r = np.square(f-c1) - np.square(f-c2)
    return r


def gs(mu, _lambda, r, u, d1, d2, b1, b2):
    d1_l = np.hstack((d1[:, 0:1], d1[:, :-1]))
    b1_l = np.hstack((b1[:, 0:1], b1[:, :-1]))
    d2_d = np.vstack((d2[0:1, :], d2[:-1, :]))
    b2_d = np.vstack((b2[0:1, :], b2[:-1, :]))
    a = d1_l - d1 - b1_l + b1 + d2_d - d2 - b2_d + b2
    u_l = np.hstack((u[:, 0:1], u[:, :-1]))
    u_r = np.hstack((u[:, 1:], u[:, -1:]))
    u_d = np.vstack((u[0:1, :], u[:-1, :]))
    u_u = np.vstack((u[1:, :], u[-1:, :]))
    beta = (u_l+u_r+u_d+u_u-mu/_lambda*r+a)/4
    u = np.maximum(np.minimum(beta, 1), 0)
    return u


def shrink_alone(d, _lambda , g):
    d1 = np.abs(d)
    return np.maximum(d1-_lambda/g, 0)*d/(d1+1e-10)


def shrink(d1, d2, _lambda, g):
    d = np.sqrt(np.square(d1) + np.square(d2))
    temp = np.maximum(d - _lambda/g, 0) / (d+1e-18)
    return temp*d1, temp*d2


def grad(u):
    u_l = np.hstack((u[:, 0:1], u[:, :-1]))
    u_d = np.vstack((u[0:1, :], u[:-1, :]))
    return u-u_l, u-u_d




def k_curve_h1(b):
    uxx = b[1:-1, 2:] + b[1:-1, :-2] - 2*b[1:-1, 1:-1]
    uyy = b[2:, 1:-1] + b[:-2, 1:-1] - 2*b[1:-1, 1:-1]
    ux = (b[1:-1, 2:] - b[1:-1, :-2])/2
    uy = (b[2:, 1:-1] - b[:-2, 1:-1])/2
    b = np.zeros([ux.shape[0]+2, ux.shape[1]+2])
    b[1:-1, 1:-1] = ux.copy()
    b[0, :] = b[1, :].copy()
    b[-1, :] = b[-2, :].copy()
    b[:, 0] = b[:, 1].copy()
    b[:, -1] = b[:, -2].copy()
    uxy = (b[2:, 1:-1]-b[:-2, 1:-1])/2
    k = ((np.square(ux)*uyy)+(np.square(uy)*uxx)-(2*ux*uy*uxy)) / \
        (np.square(ux)+np.square(uy)+1e-8)
    return k


def reinitial_2d(phi, steps):
    m, n = phi.shape
    h = 1/(max(m, n)-1)
    eps = 1e-9
    cfl = 0.5
    dt = cfl*h
    dx = 1/(m-1)
    dy = 1/(n-1)
    for k in range(steps):
        b = np.zeros([phi.shape[0]+2, phi.shape[1]+2])
        b[1:-1, 1:-1] = phi.copy()
        b[0, :] = b[1, :].copy()
        b[-1, :] = b[-2, :].copy()
        b[:, 0] = b[:, 1].copy()
        b[:, -1] = b[:, -2].copy()
        yp = (b[2:, 1:-1] - phi)/dx
        yn = (phi - b[:-2, 1:-1])/dx
        xp = (b[1:-1, 2:] - phi)/dy
        xn = (phi - b[1:-1, :-2])/dy
        phi_p = phi >= 0
        phi_n = 1-phi_p
        godnov_p = np.sqrt(np.maximum(np.square(np.maximum(xn, 0)), np.square(np.minimum(xp, 0)))
                           + np.maximum(np.square(np.maximum(yn, 0)), np.square(np.minimum(yp, 0))))
        godnov_n = np.sqrt(np.maximum(np.square(np.minimum(xn, 0)), np.square(np.maximum(xp, 0)))
                           + np.maximum(np.square(np.minimum(yn, 0)), np.square(np.maximum(yp, 0))))
        phi = phi - dt*phi_p*(godnov_p-1)*phi/np.sqrt(np.square(phi)+np.square(godnov_p)*dx*dy+eps)\
            - dt*phi_n*(godnov_n-1)*phi/np.sqrt(np.square(phi)+np.square(godnov_n)*dx*dy+eps)
    return phi


