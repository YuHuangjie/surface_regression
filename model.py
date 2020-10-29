import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

RAW_INPUT_SIZE = 3
RAW_INPUT_VIEW_SIZE = 3
RAW_INPUT_SCALE = 1/1

# Fourier feature mapping
def input_mapping(x, B): 
    if B is None:
        return x
    else:
        x_proj = torch.matmul(2.*np.pi*x, B.T)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MLP(nn.Module):
    def __init__(self, D, W, B, B_view):
        super(MLP, self).__init__()
        self.D = D
        self.W = W
        self.B = B
        self.B_view = B_view
        self.input_size = 2 * B.shape[0] if B is not None else RAW_INPUT_SIZE
        self.input_size_view = 2 * B_view.shape[0] if B_view is not None else RAW_INPUT_VIEW_SIZE
        
        self.linears = nn.ModuleList(
            [nn.Linear(self.input_size, W)] + [nn.Linear(W, W) for i in range(D-1)])
        self.view_linears = nn.ModuleList(
            [nn.Linear(W+self.input_size_view, W//2)])
        self.output_linear = nn.Linear(W//2, 3)
        
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_size, self.input_size_view], dim=-1)
        h = input_pts
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
        h = torch.cat([h, input_views], -1)
        for i, l in enumerate(self.view_linears):
            h = self.view_linears[i](h)
            h = F.relu(h)
        outputs = self.output_linear(h)
        return outputs
    
def make_network(D, W, B=None, B_view=None):
        return MLP(D, W, B, B_view).float()

model_pred = lambda model, x: model(
        torch.cat([input_mapping(x[:,:,:RAW_INPUT_SIZE]*RAW_INPUT_SCALE, model.B), 
                   input_mapping(x[:,:,RAW_INPUT_SIZE:], model.B_view)], -1))
model_loss = lambda pred, y: .5 * torch.mean((pred - y) ** 2)
model_loss2 = lambda model, x, y: .5 * torch.mean((model_pred(model, x) - y) ** 2)
model_psnr = lambda loss : -10. * torch.log10(2.*loss)