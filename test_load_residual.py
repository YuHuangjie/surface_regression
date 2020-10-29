import torch
import numpy as np
from tqdm import tqdm as tqdm
import os
import imageio

from approx_res import ApproxResSurfaceDataset as Dataset
from model import *
from rd_wrapper import rd_wrapper

device = torch.device('cuda', 1)

# Load network 
data_dir = 'data/tangsancai'
exp_dir = 'exp/tangsancai'

# Initialize training dataset
obj_path = f'{data_dir}/obj/mesh_sh2.obj'
dsize = [400, 640]
rdwrapper = rd_wrapper(*dsize, obj_path)

partition = {'train': [f'{i}' for i in range(530)], 
             'test': [f'{i}' for i in range(530, 620)],
             'video': [f'{i}' for i in range(825, 1127)]}
params = {'shuffle': False,
          'num_workers': 1}
train_set = Dataset(data_dir, obj_path, partition['train'], 'profile.txt', need_residual=True)
train_generator = torch.utils.data.DataLoader(train_set, **params)
test_set = Dataset(data_dir, obj_path, partition['test'], 'profile.txt', need_residual=True)
test_generator = torch.utils.data.DataLoader(test_set, **params)

# Compute residual training data
output_dir = f'{exp_dir}/approx_residual'
os.makedirs(output_dir, exist_ok=True)

print("Generating training residuals")
with torch.no_grad():
        for i, (x, residual, mask, approx) in enumerate(tqdm(train_generator)):
                with open(os.path.join(output_dir, f'{i}.npy'), 'wb') as f:
                        x[0, :, :3] *= RAW_INPUT_SCALE
                        np.save(f, x)
                        np.save(f, residual)
                        np.save(f, mask)

print("Generating testing residuals")
with torch.no_grad():
        for i, (x, residual, mask, approx) in enumerate(tqdm(test_generator)):
                with open(os.path.join(output_dir, f'test{i}.npy'), 'wb') as f:
                        x[0, :, :3] *= RAW_INPUT_SCALE
                        np.save(f, x)
                        np.save(f, residual)
                        np.save(f, mask)