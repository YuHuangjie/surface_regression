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
from load_approx_res import ApproxResSurfaceDataset as Dataset
from model import *

device = torch.device('cuda', 1)

obj_path = 'data/lucy/lucy-sh.obj'
dsize = [1024, 1024]
data_dir = 'data/lucy'
exp_dir = 'exp/lucy'

# Set up training/testing data
partition = {'train': [f'./train/r_{i}' for i in range(800)], 
             'test': [f'./test/r_{i}' for i in range(200)]}
train_params = {'shuffle': True,
          'num_workers': 1,}
test_params = {'shuffle': False,
          'num_workers': 1,}

training_set = Dataset('blender', data_dir, obj_path, partition['train'], 'transforms_train.json', need_residual=True)
training_generator = torch.utils.data.DataLoader(training_set, **train_params)

test_set = Dataset('blender', data_dir, obj_path, partition['test'], 'transforms_test.json', need_residual=True)
test_generator = torch.utils.data.DataLoader(test_set, **test_params)

def train_model(D, W, learning_rate, epochs, B, B_view, training_generator):

    model = make_network(D, W, B, B_view).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_psnrs = []
    xs = []
    for i in tqdm(range(epochs), desc='epoch', leave=False):
        total_loss = 0
        for local_batch, local_labels, _, _ in training_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Forward pass
            loss = model_loss2(model, local_batch, local_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        if i==0 or i % 100 != 0:
            continue
        train_psnrs.append(model_psnr(torch.Tensor([total_loss/len(training_generator)])))
        xs.append(i)

    return {
        'train_psnrs': train_psnrs,
        'xs': xs,
        'model': model,
    }

network_size = (8, 256)
learning_rate = 1e-4
epochs = 1500
posenc_scale = 6
mapping_size = 256
mapping_size_view = 256
gauss_scale = [10.]
gauss_scale_view = [0.2]

include_none = True
include_basic = False
include_pe = False
include_gauss = False

B_dict = {}
B_view_dict = {}
if include_none:
    # Standard network - no mapping
    B_dict['none'] = None
    B_view_dict['none'] = None
if include_basic:
    # Basic mapping
    B_dict['basic'] = torch.eye(RAW_INPUT_SIZE).to(device)
    B_view_dict['basic'] = torch.eye(RAW_INPUT_VIEW_SIZE).to(device)
if include_pe:
    # Positional encoding
    B_pe = 2**torch.linspace(0, posenc_scale, posenc_scale + 1)
    B_pe = torch.cat([B_pe.reshape((B_pe.shape[0],1)), torch.zeros((B_pe.shape[0], RAW_INPUT_SIZE-1))], 1)
    b = B_pe
    for i in range(RAW_INPUT_SIZE-1):
        B_pe = torch.cat([B_pe, torch.roll(b, i+1, dims=-1)], 0)
    B_dict['pe'] = B_pe.to(device)
    
    B_pe = 2**torch.linspace(0, posenc_scale, posenc_scale + 1)
    B_pe = torch.cat([B_pe.reshape((B_pe.shape[0],1)), torch.zeros((B_pe.shape[0], RAW_INPUT_VIEW_SIZE-1))], 1)
    b = B_pe
    for i in range(RAW_INPUT_VIEW_SIZE-1):
        B_pe = torch.cat([B_pe, torch.roll(b, i+1, dims=-1)], 0)
    B_view_dict['pe'] = B_pe.to(device)
if include_gauss:
    B_gauss = torch.normal(0, 1, size=(mapping_size, RAW_INPUT_SIZE)).to(device)
    B_gauss_view = torch.normal(0, 1, size=(mapping_size_view, RAW_INPUT_VIEW_SIZE)).to(device)
    for scale in gauss_scale_view:
        B_dict[f'gauss_{scale}'] = B_gauss * gauss_scale[0]
        B_view_dict[f'gauss_{scale}'] = B_gauss_view * scale

# This should take about 2-3 minutes
outputs = {}
for k in tqdm(B_dict):
    outputs[k] = train_model(*network_size, learning_rate, epochs, B_dict[k], B_view_dict[k], 
                             training_generator)

    # Make and save predicted images
    model = outputs[k]['model']
    output_dir = f'{exp_dir}/{k}/test'
    os.makedirs(output_dir, exist_ok=True)
    outputs[k]['test_psnrs'] = []

    with torch.no_grad():
        for i, (x, residual, mask, approx) in enumerate(test_generator):
            x, residual = x.to(device), residual.to(device)
            y = model_pred(model, x)
            img = torch.zeros((dsize[0]*dsize[1], 3))
            img[mask[0]] = y.cpu() + approx[0].cpu()
            img = np.clip(img.numpy() * 255., 0, 255).astype(np.uint8)
            img = img.reshape(dsize + [3])

            imageio.imwrite(f'{output_dir}/{i}.png', img)
            outputs[k]['test_psnrs'].append(model_psnr(model_loss(y, residual)).cpu().numpy())

    # save model
    save_dict = model.state_dict()
    save_dict['D'] = model.D
    save_dict['W'] = model.W
    save_dict['B'] = model.B
    save_dict['B_view'] = model.B_view
    torch.save(save_dict, f'{output_dir}/model.pt')

    # save test psnrs
    np.save(f'{output_dir}/test_psnr.npy', outputs[k]['test_psnrs'])

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
    plt.plot(outputs[k]['test_psnrs'], label=k)
plt.title('Test error')
plt.ylabel('PSNR')
plt.xlabel('Training iter')
plt.legend()

plt.savefig(f'{exp_dir}/residual-regression-psnr.png')