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
            K = exp(-||x_a-x_a'||*a_pos) * max(0, <x_b, x_b'>)^a_dir
    
    Note that the positional component has a close-form Fourier transform.
    Check Wikipedia if you don't remember.

    We first take the Fourier transform of K and regard it as a (unnormalized)
    probability distribution. We then sample from it.
    '''
    frq_r_pos = np.linspace(0, 100, 21) * 10
    power = 1
    for i, fr in enumerate(frq_r_pos):
        c = np.array([fr, a_pos, power])
        user_data = cast(c.ctypes.data_as(POINTER(c_double)), c_void_p)
        func = LowLevelCallable(lib.integrand_pos, user_data)
        if integrate.nquad(func, [[-2, 2], [-2, 2], [-2, 2]])[0] < 1e-5:
            break
    print(frq_r_pos[i])
    frq_r_pos = np.linspace(0, frq_r_pos[i], int(frq_r_pos[i]) if int(frq_r_pos[i]) > 40 else 40)
    freq_pos = np.zeros_like(frq_r_pos)
    for i, fr in enumerate(frq_r_pos):
        c = np.array([fr, a_pos, power])
        user_data = cast(c.ctypes.data_as(POINTER(c_double)), c_void_p)
        func = LowLevelCallable(lib.integrand_pos, user_data)
        freq_pos[i] = integrate.nquad(func, [[-2, 2], [-2, 2], [-2, 2]])[0]
    ###
    ###
    frq_r_dir = np.linspace(0, 100, 21) * 10
    for i, fr in enumerate(frq_r_dir):
        c = np.array([fr, a_dir])
        user_data = cast(c.ctypes.data_as(POINTER(c_double)), c_void_p)
        func = LowLevelCallable(lib.integrand_dir, user_data)
        if integrate.nquad(func, [[-2, 2], [-2, 2], [-2, 2]])[0] < 1e-5:
            break
    print(frq_r_dir[i])
    frq_r_dir = np.linspace(0, frq_r_dir[i], int(frq_r_dir[i]) if int(frq_r_dir[i]) > 40 else 40)
    freq_dir = np.zeros_like(frq_r_dir)
    for i, fr in enumerate(frq_r_dir):
        c = np.array([fr, a_dir])
        user_data = cast(c.ctypes.data_as(POINTER(c_double)), c_void_p)
        func = LowLevelCallable(lib.integrand_dir, user_data)
        freq_dir[i] = integrate.nquad(func, [[-2, 2], [-2, 2], [-2, 2]])[0]

    # perform rejection sampling
    samples = np.zeros((N*2, 6))
    i = 0

    print('perform rejection sampling')
    
    while i < N:
        x,y,z = np.random.uniform(-frq_r_pos[-1], frq_r_pos[-1], (3, N*10))
        dx,dy,dz = np.random.uniform(-frq_r_dir[-1], frq_r_dir[-1], (3, N*10))
        p = np.random.uniform(0, freq_pos[0]*freq_dir[0], N*10)
        u = np.interp((x*x+y*y+z*z)**0.5, frq_r_pos, freq_pos, right=0) \
            * np.interp((dx*dx+dy*dy+dz*dz)**0.5, frq_r_dir, freq_dir, right=0)

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