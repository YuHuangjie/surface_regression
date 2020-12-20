import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CustomizedLinear import CustomizedLinear

class NoMap:
    def __init__(self):
        pass
    def map_size(self):
        return 6
    def map(self, X):
        return X

class FFM:
    def __init__(self, B):
        def proj(x, B):
            x_proj = torch.matmul(x, B.T)
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        self.B = torch.Tensor(B)
        self.map_f = lambda x: proj(x, self.B)

    def map_size(self):
        return 2*self.B.shape[0]

    def map(self, X):
        if self.B.device != X.device:
            self.B = self.B.to(X.device)
        return self.map_f(X)

class RFF:
    '''
    Generate Random Fourier Features (RFF) corresponding to the kernel:
        k(x, y) = k_a(x_a, y_a)*k_b(x_b, y_b)
    where 
        k_a(x_a, y_a) = exp(-\norm(x_a-y_a)/gamma_1),
        k_b(x_b, y_b) = <x_b, y_b>^gamma_2.
    '''
    def __init__(self, W, b):
        self.W = torch.Tensor(W)
        self.b = torch.Tensor(b)
        self.N = self.W.shape[0]

    def map_size(self):
        return self.N

    def map(self, X):
        if self.W.device != X.device:
            self.W = self.W.to(X.device)
            self.b = self.b.to(X.device)
        Z = torch.cos(X @ self.W.T + self.b)
        return Z

class MLP(nn.Module):
    def __init__(self, D, W, map):
        super(MLP, self).__init__()
        self.D = D
        self.W = W
        self.map = map
        self.input_size = map.map_size()
        
        self.linears = nn.ModuleList(
            [nn.Linear(self.input_size, W)] + [nn.Linear(W, W) for i in range(D)])
        self.output_linear = nn.Linear(W, 3)

    def forward(self, x):
        h = self.map.map(x)
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
        outputs = self.output_linear(h)
        return outputs


class PMLP(nn.Module):
    def __init__(self, D, W, map, degree=1.0):
        super(PMLP, self).__init__()
        self.D = D
        self.W = W
        self.map = map
        self.input_size = map.map_size()

        linears = []
        layer_shape = [self.input_size] + [W]*(D+1)
        for i in range(len(layer_shape)-2):
            degree = degree if i==0 else 1.0
            mask = self.build_graph(layer_shape[i], layer_shape[i+1], degree=int(layer_shape[i+1]*degree))
            linears += [CustomizedLinear(mask)]
        self.output_linear = nn.Linear(W, 3)
        self.linears = nn.Sequential(*linears)

    def build_graph(self, in_dim, out_dim, degree=0):
        mask = torch.zeros(in_dim, out_dim)
        start = np.arange(in_dim)
        end = start + out_dim
        for i in range(in_dim):
            ind = np.mod(np.linspace(start[i], end[i], degree, endpoint=False).astype('int'), out_dim)
            mask[i, ind] = 1
        return mask>0

    def forward(self, x):
        B, N = x.shape[:2]
        h = self.map.map(x)
        for i, l in enumerate(self.linears):
            h = self.linears[i](h.view(B*N,-1))
            h = F.relu(h)
        outputs = self.output_linear(h).view(B,N,-1)
        return outputs

def make_ffm_network(D, W, B=None):
    map = FFM(B)
    return MLP(D, W, map).float()

def make_rff_network(D, W, We, b):
    map = RFF(We, b)
    return MLP(D, W, map).float()

def make_prff_network(D, W, We, b, degree):
    map = RFF(We, b)
    return PMLP(D, W, map, degree=degree).float()

def make_relu_network(D, W):
    map = NoMap()
    return MLP(D, W, map).float()
