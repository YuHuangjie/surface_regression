import numpy as np
from reject_sample import joint_reject_sample

class RFF:
        '''
        Generate Random Fourier Features (RFF) corresponding to the kernel:
                k(x, y) = k_a(x_a, y_a)*k_b(x_b, y_b)
        where 
                k_a(x_a, y_a) = exp(-\norm(x_a-y_a)/gamma_1),
                k_b(x_b, y_b) = <x_b, y_b>^gamma_2.
        '''
        def __init__(self, R_p, F_p, R_d, F_d, N=1000):
                self.W = joint_reject_sample(R_p=R_p, F_p=F_p, R_d=R_d, F_d=F_d, N=N)
                self.b = np.random.uniform(0, 2*np.pi, size=(N, 1))
                self.N = N

        def map(self, X):
                # let b broadcast
                Z = np.sqrt(2) / np.sqrt(self.N) * np.cos(self.W @ X.T + self.b)
                return Z
