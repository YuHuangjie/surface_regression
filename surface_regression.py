# This script mainly aims to experimenting with partial configurations (sigma=0.2)
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import os, imageio
import time

from rd_wrapper import rd_wrapper
from load_blender_surface import BlenderSurfaceDataset

device = torch.device('cuda', 0)

obj_path = 'data/lucy2/lucy.obj'
dsize = [512, 512]
data_dir = 'data/lucy2'

# Set up depth rendering engine
rdwrapper = rd_wrapper(*dsize, obj_path)

# Set up training/testing data
partition = {'train': [f'./train/r_{i}' for i in range(100)],
             'test': [f'./test/r_{i}' for i in [90]]}
params = {'shuffle': True,
          'num_workers': 1,}
training_limit=350000
training_set = BlenderSurfaceDataset(data_dir, rdwrapper, partition['train'], 'transforms_train.json', limit=training_limit, dsize=dsize)
training_generator = torch.utils.data.DataLoader(training_set, **params)
test_set = BlenderSurfaceDataset(data_dir, rdwrapper, partition['test'], 'transforms_test.json', dsize=dsize)
test_generator = torch.utils.data.DataLoader(test_set, **params)

# Fourier feature mapping
def input_mapping(x, B): 
    if B is None:
        return x
    else:
        x_proj = torch.matmul(2.*np.pi*x, B.T)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MLP(nn.Module):
    def __init__(self, D, W, input_size, input_size_view):
        super(MLP, self).__init__()
        self.D = D
        self.W = W
        self.input_size = input_size
        self.input_size_view = input_size_view
        
        self.linears = nn.ModuleList(
            [nn.Linear(input_size, W)] + [nn.Linear(W, W) for i in range(D-1)])
        self.view_linears = nn.ModuleList(
            [nn.Linear(W+input_size_view, W//2)])
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
    
def make_network(D, W, input_size, input_size_view):
    return MLP(D, W, input_size, input_size_view)

def train_model(D, W, learning_rate, epochs, B, B_view, input_dim, input_dim_view, training_generator, test_generator):
    
    model_pred = lambda model, x: model(torch.cat(
        [input_mapping(x[:,:,:input_dim], B), input_mapping(x[:,:,input_dim:], B_view)], -1))
    model_loss_pred = lambda pred, y: .5 * torch.mean((pred - y) ** 2)
    model_loss = lambda model, x, y: .5 * torch.mean((model_pred(model, x) - y) ** 2)
    model_psnr_loss = lambda loss : -10. * torch.log10(2.*loss)
    model_psnr = lambda model, x, y: -10 * torch.log10(2.*model_loss(model, x, y))
    
    input_size = 2 * B.shape[0] if B is not None else input_dim
    input_size_view = 2 * B_view.shape[0] if B_view is not None else input_dim_view
    model = make_network(D, W, input_size, input_size_view).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_psnrs = []
    test_psnrs = []
    pred_imgs = []
    xs = []
    for i in tqdm(range(epochs), desc='epoch', leave=False):
        total_loss = 0
        for local_batch, local_labels, _ in training_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Forward pass
            loss = model_loss(model, local_batch, local_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        if i==0 or i % 100 != 0:
            continue
        train_psnrs.append(model_psnr_loss(torch.Tensor([total_loss])))
        with torch.no_grad():
            pred_img = []
            for local_batch, local_labels, mask in test_generator:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                y = model_pred(model, local_batch)
                img = torch.ones((dsize[0]*dsize[1], 3))
                img[torch.squeeze(mask)] = y.cpu()
                pred_img.append(img.reshape(dsize + [3]))
            test_psnrs.append(model_psnr_loss(model_loss_pred(y, local_labels)))
            pred_imgs.append(torch.stack(pred_img))
            xs.append(i)
                
    return {
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'pred_imgs' : torch.stack(pred_imgs),
        'xs': xs,
    }

network_size = (8, 256)
# network_size_view = (1, 128)
learning_rate = 1e-4
epochs = 3001
posenc_scale = 6
mapping_size = 256
mapping_size_view = 256
gauss_scale = [10.]
gauss_scale_view = [0.2]
input_dim = 3      # [3: x-y-z]
input_dim_view = 3 # [3: dx-dy-dz] or [2: theta-phi]

include_none = False
include_basic = False
include_pe = False
include_gauss = True

B_dict = {}
B_view_dict = {}
if include_none:
    # Standard network - no mapping
    B_dict['none'] = None
    B_view_dict['none'] = None
if include_basic:
    # Basic mapping
    B_dict['basic'] = torch.eye(input_dim).to(device)
    B_view_dict['basic'] = torch.eye(input_dim_view).to(device)
if include_pe:
    # Positional encoding
    B_pe = 2**torch.linspace(0, posenc_scale, posenc_scale + 1)
    B_pe = torch.cat([B_pe.reshape((B_pe.shape[0],1)), torch.zeros((B_pe.shape[0], input_dim-1))], 1)
    b = B_pe
    for i in range(input_dim-1):
        B_pe = torch.cat([B_pe, torch.roll(b, i+1, dims=-1)], 0)
    B_dict['pe'] = B_pe.to(device)
    
    B_pe = 2**torch.linspace(0, posenc_scale, posenc_scale + 1)
    B_pe = torch.cat([B_pe.reshape((B_pe.shape[0],1)), torch.zeros((B_pe.shape[0], input_dim_view-1))], 1)
    b = B_pe
    for i in range(input_dim_view-1):
        B_pe = torch.cat([B_pe, torch.roll(b, i+1, dims=-1)], 0)
    B_view_dict['pe'] = B_pe.to(device)
if include_gauss:
    B_gauss = torch.normal(0, 1, size=(mapping_size, input_dim)).to(device)
    B_gauss_view = torch.normal(0, 1, size=(mapping_size_view, input_dim_view)).to(device)
    for scale in gauss_scale_view:
#         B_dict[f'gauss_{scale}'] = B_gauss * scale
        B_dict[f'gauss_{scale}'] = B_gauss * gauss_scale[0]
        B_view_dict[f'gauss_{scale}'] = B_gauss_view * scale

# This should take about 2-3 minutes
outputs = {}
for k in tqdm(B_dict):
    outputs[k] = train_model(*network_size, learning_rate, epochs, B_dict[k], B_view_dict[k], 
                             input_dim, input_dim_view, training_generator, test_generator)
    I = outputs[k]['pred_imgs'][-1].numpy() * 255.
    I = np.squeeze(np.clip(I, 0, 255).astype(np.uint8))
    imageio.imwrite(f'{k}-{network_size[0]}-{network_size[1]}-{dsize[0]}.jpg', I)

# Show final network outputs

plt.figure(figsize=(26,26))
N = len(outputs)
for i, k in enumerate(outputs):
    cols = 3
    rows = (N+1+cols) // cols
    plt.subplot(rows,cols,i+1)
    plt.imshow((np.clip(outputs[k]['pred_imgs'][-1,0].numpy(),0,1)*255).astype(np.uint8))
    plt.title(k)
# plt.subplot(rows,cols,N+1)
# plt.imshow(img)
# plt.title('GT')
plt.savefig('surface-regression-pred-images.png')

# Plot train/test error curves

plt.figure(figsize=(16,6))

plt.subplot(121)
for i, k in enumerate(outputs):
    plt.plot(outputs[k]['xs'], outputs[k]['train_psnrs'], label=k)
plt.title('Train error')
plt.ylabel('PSNR')
plt.xlabel('Training iter')
plt.legend()

plt.subplot(122)
for i, k in enumerate(outputs):
    plt.plot(outputs[k]['xs'], outputs[k]['test_psnrs'], label=k)
plt.title('Test error')
plt.ylabel('PSNR')
plt.xlabel('Training iter')
plt.legend()

plt.savefig('surface-regression-psnr.png')