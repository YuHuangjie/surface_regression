from scipy.integrate import tplquad
import numpy as np
import matplotlib.pyplot as plt
from reject_sample import reject_sample

def integrand_dir(z, y, x, r, gamma=0.01):
    r = r / (3**0.5)
    t = 1-0.5*(x*x+y*y+z*z)
    return (0 if t < 0 else t)**gamma * np.exp(-1j*r*(x+y+z))

def integrand_pos(z, y, x, r, gamma = 10):
    r = r / (3**0.5)
    return np.exp(-(x*x+y*y+z*z)**0.5/gamma) * np.exp(-1j*r*(x+y+z))

kernel = 'pkernel'      # 'pkernel' or 'dkernel'
regenerate = False  # True or False

if regenerate:
    if kernel == 'pkernel':
        R = np.linspace(0, 50, 100)
        gamma = 0.1
        integrand = integrand_pos
    elif kernel == 'dkernel':
        R = np.linspace(0, 15, 100)
        gamma = 10
        integrand = integrand_dir
    F = np.zeros_like(R)

    for i, r in enumerate(R):
        print(i, end=', ')
        F[i] = tplquad(integrand, -2, 2, lambda x:-2, lambda x:2, lambda x,y:-2, lambda x,y:2, args=(gamma, r))[0]
    
    np.save(f'{kernel}_spectrum.npy', (R, F))
else:
    (R, F) = np.load(f'{kernel}_spectrum.npy')

# do rejection sampling
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(8, 4)
axes[0].plot(R, F)
axes[0].set_title('Exact Spectrum')
axes[0].set_xticks([])
axes[0].set_yticks([])

samples = reject_sample(R, F, 10000)

axes[1].hist(np.linalg.norm(samples, axis=1), bins=100)
axes[1].set_title('rejection sampling')
plt.tight_layout()
plt.show()
