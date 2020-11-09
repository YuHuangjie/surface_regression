import matplotlib.pyplot as plt
import numpy as np
from   sklearn.metrics.pairwise import rbf_kernel
from   sklearn.datasets import make_s_curve
from reject_sample import reject_sample, joint_reject_sample

def pkernel(X, gamma):
    K = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        K[i, :] = np.exp(-1*np.linalg.norm(X[i,:]-X, axis=1)/gamma)
    return K

def dkernel(X, gamma):
    K = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        K[i, :] = np.maximum(0., 1-0.5*np.linalg.norm(X[i,:]-X, axis=1)**2) ** gamma
    return K

def joint_kernel(X, pgamma, dgamma):
    pK = pkernel(X[:, :3], pgamma)
    dK = dkernel(X[:, 3:], dgamma)
    return pK * dK

def pkernel_X(N):
    s = np.linspace(0, 0.1, int(np.cbrt(N)))
    X,Y,Z = np.meshgrid(s, s, s)
    X = np.stack([X.reshape((-1)), Y.reshape((-1)), Z.reshape((-1))], axis=1)
    X = X[np.argsort(np.linalg.norm(X, axis=1))]
    return X

def dkernel_X(N):
    X = np.random.randn(N, 3)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X[np.argsort(np.arccos(X[:, 2]))]
    return X

# -----------------------------------------------------------------------------
N    = 1000
kernel = 'joint_kernel'
D    = 6 if kernel=='joint_kernel' else 3

if kernel == 'pkernel':
    X = pkernel_X(N)
    K    = pkernel(X, gamma=0.1)
    (R_p, F_p) = np.load('pkernel_spectrum.npy')
elif kernel == 'dkernel':
    X = dkernel_X(N)
    K = dkernel(X, gamma=10)
    (R_d, F_d) = np.load('dkernel_spectrum.npy')
elif kernel == 'joint_kernel':
    X = np.hstack([pkernel_X(N), dkernel_X(N)])
    K = joint_kernel(X, pgamma=0.1, dgamma=10)
    (R_d, F_d) = np.load('dkernel_spectrum.npy')
    (R_p, F_p) = np.load('pkernel_spectrum.npy')

fig, axes = plt.subplots(1, 5)
fig.set_size_inches(15, 4)

axes[0].imshow(K, cmap=plt.cm.Blues)
axes[0].set_title('Exact kernel')
axes[0].set_xticks([])
axes[0].set_yticks([])

for R, ax in zip([10, 100, 1000, 10000], axes[1:]):
    if kernel == 'joint_kernel':
        W = joint_reject_sample(R_p=R_p, F_p=F_p, R_d=R_d, F_d=F_d, N=R)
    elif kernel == 'pkernel':
        W = reject_sample(R_p, F_p, R)
    else:
        W = reject_sample(R_d, F_d, R)

    b    = np.random.uniform(0, 2*np.pi, size=R)
    B    = np.repeat(b[:, np.newaxis], N, axis=1)
    Z    = np.sqrt(2) / np.sqrt(R) * np.cos(W @ X.T + B)
    ZZ   = Z.T@Z
    ax.imshow(ZZ, cmap=plt.cm.Blues)
    ax.set_title(r'$\mathbf{Z} \mathbf{Z}^{\top}$, $R=%s$' % R)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
