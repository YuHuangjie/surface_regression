import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import os, imageio
import time
import cv2

from rd_wrapper import rd_wrapper
from load_approx_res import ApproxResSurfaceDataset as Dataset
from model import *
from rff.reject_sample import *

device = torch.device('cuda', 3)

datatype = 'blender'    # 'blender' or 'dslf'
expname = 'greenstone_down'     # dataset identifier
maxL = 3                # maximum order of SH basis
test_only = False
dsize = [512, 512]
obj_path = f'data/{expname}/{expname}-sh.obj'
data_dir = f'data/{expname}'
exp_dir = f'exp/{expname}-L{maxL}'

# Set up training/testing data
partition = {'train': [f'./train/r_{i}' for i in range(800)], 
             'test': [f'./test/r_{i}' for i in range(200)]}
train_params = {'shuffle': True,
          'num_workers': 1,}
test_params = {'shuffle': False,
          'num_workers': 1,}

training_set = Dataset(datatype, data_dir, obj_path, partition['train'], 
                    'transforms_train.json', need_residual=True, L=maxL)
training_generator = torch.utils.data.DataLoader(training_set, **train_params)

test_set = Dataset(datatype, data_dir, obj_path, partition['test'], 
                    'transforms_test.json', need_residual=True, L=maxL)
test_generator = torch.utils.data.DataLoader(test_set, **test_params)

def train_model(D, W, maptype, learning_rate, epochs, training_generator, test_only, *map_params):

    if maptype == 'ffm':
        model = make_ffm_network(D, W, *map_params).to(device)
    elif maptype == 'rff':
        model = make_rff_network(D, W, *map_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if test_only:
        return {'model': model}

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

        if (i % 100 == 0 and i != 0) or i == epochs-1:
            train_psnrs.append(model_psnr(torch.Tensor([total_loss/len(training_generator)])))
            xs.append(i)
            print(train_psnrs[-1])

    return {
        'train_psnrs': train_psnrs,
        'xs': xs,
        'model': model,
    }

network_size = (8, 256)
learning_rate = 1e-4
epochs = 100

posenc_scale = 6

mapping_size = 2048
mapping_size_view = 2048
gauss_scale = 8
gauss_scale_view = 0.6

rff_size = 8192

include_none = False
include_basic = False
include_pe = False
include_gauss = True
include_rff = True

map_params = {}
map_type = {}
if include_none:
    # Standard network - no mapping
    map_params['none'] = (None, None)
    map_type['none'] = 'ffm'
if include_basic:
    # Basic mapping
    map_params['basic'] = (torch.eye(3).to(device), torch.eye(3).to(device))
    map_type['basic'] = 'ffm'
if include_pe:
    # Positional encoding
    B = 2**torch.linspace(0, posenc_scale, posenc_scale + 1)
    B = torch.cat([B.reshape((B.shape[0],1)), torch.zeros((B.shape[0], 3-1))], 1)
    b = B
    for i in range(3-1):
        B = torch.cat([B, torch.roll(b, i+1, dims=-1)], 0)
    
    B_view = 2**torch.linspace(0, posenc_scale, posenc_scale + 1)
    B_view = torch.cat([B_view.reshape((B_view.shape[0],1)), torch.zeros((B_view.shape[0], 3-1))], 1)
    b = B_view
    for i in range(3-1):
        B_view = torch.cat([B_view, torch.roll(b, i+1, dims=-1)], 0)
    map_params['pe'] = (B.to(device), B_view.to(device))
    map_type['pe'] = 'ffm'
if include_gauss:
    B_gauss = torch.normal(0, 1, size=(mapping_size, 3)).to(device)
    B_gauss_view = torch.normal(0, 1, size=(mapping_size_view, 3)).to(device)
    map_params['ffm'] = (B_gauss * gauss_scale, B_gauss_view * gauss_scale_view)
    map_type['ffm'] = 'ffm'
if include_rff:
    (R_p, F_p) = np.load('rff/pkernel_spectrum_0.001.npy')
    (R_d, F_d) = np.load('rff/dkernel_spectrum_6.npy')
    W = joint_reject_sample(R_p=R_p, F_p=F_p, R_d=R_d, F_d=F_d, N=rff_size)
    b = np.random.uniform(0, 2*np.pi, size=(1, rff_size))
    map_params['rff'] = (W, b)
    map_type['rff'] = 'rff'

if test_only:
    # override mapping parameters 
    for k in map_params:
        map_params[k] = torch.load(f'{exp_dir}/{k}/test/model.pt')['map_params']

# This should take about 2-3 minutes
outputs = {}
for k in tqdm(map_params):
    outputs[k] = train_model(*network_size, map_type[k], learning_rate, epochs, 
                        training_generator, test_only, *map_params[k])

    model = outputs[k]['model']
    output_dir = f'{exp_dir}/{k}/test'
    os.makedirs(output_dir, exist_ok=True)
    print(f'writing output to {output_dir}')

    if not test_only:
        # save model
        save_dict = model.state_dict()
        save_dict['D'] = model.D
        save_dict['W'] = model.W
        save_dict['map_type'] = map_type[k]
        save_dict['map_params'] = map_params[k]
        torch.save(save_dict, f'{output_dir}/model.pt')
    else:
        # load model
        save_dict = torch.load(f'{output_dir}/model.pt')
        model.load_state_dict(save_dict, strict=False)

    # Make and save predicted images
    outputs[k]['test_psnrs'] = []

    with torch.no_grad():
        for i, (x, residual, mask, approx) in enumerate(test_generator):
            x, residual = x.to(device), residual.to(device)
            y = model_pred(model, x)
            img = torch.zeros((dsize[0]*dsize[1], 3))
            img[mask[0]] = y.cpu() + approx[0].cpu()
            img = np.clip(img.numpy() * 255., 0, 255).astype(np.uint8)
            img = img.reshape(dsize + [3])

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assert(cv2.imwrite(f'{output_dir}/{i}.png', img))
            outputs[k]['test_psnrs'].append(model_psnr(model_loss(y, residual)).cpu().numpy())

    # save test psnrs
    np.save(f'{output_dir}/test_psnr.npy', outputs[k]['test_psnrs'])

# Plot train/test error curves

plt.figure(figsize=(16,6))

if not test_only:
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