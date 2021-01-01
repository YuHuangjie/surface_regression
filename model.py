import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class DSLF(nn.Module):
    def __init__(self):
        super(DSLF, self).__init__()

        self.FC_1 = nn.ModuleList([nn.Linear(3, 512)] + [nn.Linear(512, 256)])
        self.FC_2 = nn.ModuleList([nn.Linear(3, 512)] + [nn.Linear(512, 256)])
        self.final = nn.ModuleList([nn.Linear(512, 1024)] + [nn.Linear(1024, 256)] + \
                                   [nn.Linear(256, 128)])
        self.output = nn.Linear(128, 3)

    def forward(self, x):
        h1 = x[..., :3]
        for l in self.FC_1:
            h1 = l(h1)
            h1 = F.relu(h1)
        h2 = x[..., 3:]
        for l in self.FC_2:
            h2 = l(h2)
            h2 = F.relu(h2)
        h = torch.cat((h1, h2), dim=-1)
        for l in self.final:
            h = l(h)
            h = F.relu(h)
        outputs = self.output(h)

        return outputs
    
def make_ffm_network(D, W, B=None):
    map = FFM(B)
    return MLP(D, W, map).float()

def make_rff_network(D, W, We, b):
    map = RFF(We, b)
    return MLP(D, W, map).float()

def make_relu_network(D, W):
    map = NoMap()
    return MLP(D, W, map).float()

def make_dslf_network():
    return DSLF().float()
