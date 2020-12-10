import numpy as np
import os
from ctypes import *
from scipy import integrate, LowLevelCallable

def slf_sample_init(lib_path):
    lib = CDLL(os.path.abspath(lib_path))
    lib.integrand_pos.restype = c_double
    lib.integrand_pos.argtypes = (c_int, POINTER(c_double), c_void_p)

    lib.integrand_dir.restype = c_double
    lib.integrand_dir.argtypes = (c_int, POINTER(c_double), c_void_p)

    return lib

def slf_sample(lib, a_pos, a_dir, N):
    '''
    Return random frequencies as in Random Fourier Features.

    The kernel is defined as:
            K = exp(-||x_a-x_a'||^2*a_pos) * max(0, <x_b, x_b'>)^a_dir

    We first take the Fourier transform of K and regard it as a (unnormalized)
    probability distribution. We then sample from it.
    '''
    # First determine the range of radius. Too large radius leads to very small
    # frequency, and therefore difficult to apply rejection sampling.
    radius = np.linspace(0, 20, 20) * 10
    for i, r in enumerate(radius):
        c = np.array([r, a_pos])
        user_data = cast(c.ctypes.data_as(POINTER(c_double)), c_void_p)
        func = LowLevelCallable(lib.integrand_pos, user_data)
        if integrate.nquad(func, [[-2, 2], [-2, 2], [-2, 2]])[0] < 1e-5:
            break
    R_p = np.linspace(0, radius[i], int(radius[i]) if int(radius[i]) > 100 else 100)

    for i, r in enumerate(radius):
        c = np.array([r, a_dir])
        user_data = cast(c.ctypes.data_as(POINTER(c_double)), c_void_p)
        func = LowLevelCallable(lib.integrand_dir, user_data)
        if integrate.nquad(func, [[-2, 2], [-2, 2], [-2, 2]])[0] < 1e-5:
            break
    R_d = np.linspace(0, radius[i], int(radius[i]) if int(radius[i]) > 100 else 100)

    print((R_p[-1], R_d[-1]))

    # Take the fourier transform of positional kernel. Store the transform as a
    # function of the norm of frequency.
    F_p = np.zeros_like(R_p)
    for i, r in enumerate(R_p):
        c = np.array([r, a_pos])
        user_data = cast(c.ctypes.data_as(POINTER(c_double)), c_void_p)
        func = LowLevelCallable(lib.integrand_pos, user_data)
        F_p[i] = integrate.nquad(func, [[-2, 2], [-2, 2], [-2, 2]])[0]

    # Take the fourier transform of directional kernel. Store the transform as a
    # function of the norm of frequency.
    F_d = np.zeros_like(R_d)
    for i, r in enumerate(R_d):
        c = np.array([r, a_dir])
        user_data = cast(c.ctypes.data_as(POINTER(c_double)), c_void_p)
        func = LowLevelCallable(lib.integrand_dir, user_data)
        F_d[i] = integrate.nquad(func, [[-2, 2], [-2, 2], [-2, 2]])[0]

    # perform rejection sampling
    samples = np.zeros((N*2, 6))
    i = 0

    while i < N:
        x,y,z = np.random.uniform(-R_p[-1], R_p[-1], (3, N))
        dx,dy,dz = np.random.uniform(-R_d[-1], R_d[-1], (3, N))
        p = np.random.uniform(0, F_p[0]*F_d[0], N)
        u = np.interp((x*x+y*y+z*z)**0.5, R_p, F_p, right=0) * np.interp((dx*dx+dy*dy+dz*dz)**0.5, R_d, F_d, right=0)

        mask = p < u
        if mask.sum() > 0:
            samples[i:i+mask.sum()] = np.hstack([
                    x[mask].reshape((-1,1)), 
                    y[mask].reshape((-1,1)), 
                    z[mask].reshape((-1,1)), 
                    dx[mask].reshape((-1,1)), 
                    dy[mask].reshape((-1,1)), 
                    dz[mask].reshape((-1,1))])
            i += mask.sum()
    return samples[:N]