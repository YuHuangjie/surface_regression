import os
import torch
import numpy as np
import imageio 
import cv2
import time
from ctypes import *
from tqdm import tqdm as tqdm

from dslf_util import load_dslf_config
from blender_util import load_blender_config
from rd_wrapper import rd_wrapper
from rsh_wrapper import rsh_wrapper

class ApproxResSurfaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, datadir, objmesh, list_ids, profile_file, need_residual=False, dsize=None, L=-1):
        super(ApproxResSurfaceDataset).__init__()

        self.list_ids = list_ids

        self.x_trains = {}
        self.masks = {}
        self.labels = {}
        self.residuals = {}
        self.approx = {}
        self.need_residual = need_residual

        if dataset == 'dslf':
            self.list_c2ws, self.list_Ks, self.list_images = load_dslf_config(
                datadir, profile_file)
        elif dataset == 'blender':
            self.list_c2ws, self.list_Ks, self.list_images = load_blender_config(
                datadir, profile_file)
        else:
            raise RuntimeError(f'Unknown dataset type {dataset}')

        # read ground-truths
        if need_residual:
            print('Stage: loading images')
            for id in tqdm(list_ids):
                imagepath = self.list_images[id]
                I = imageio.imread(imagepath, pilmode='RGB')
                I = (np.array(I) / 255.).astype(np.float32)
                self.H, self.W = I.shape[:2]
                self.labels[id] = I.reshape((-1, 3))
        else:
            print('Pass loading ground truths')
            self.H = dsize[0]
            self.W = dsize[1]

        # render depth maps
        print('Stage: loading depths')
        rdwrapper = rd_wrapper(self.H, self.W, objmesh)
        self.depths = {}
        for id in tqdm(list_ids):
            c2w = self.list_c2ws[id]
            K = self.list_Ks[id]
            depth = np.zeros((self.H * self.W, 1), dtype=np.float32)
            rdwrapper.render_depth(c2w, K, depth)
            self.depths[id] = depth

        # calculate x-y-z-theta-phi and mask
        print('Stage: loading x-y-z-theta-phi')
        for id in tqdm(list_ids):
            self.x_trains[id], self.masks[id] = self.calculate_x_y_z_theta_phi(id)
        del self.depths

        # render 'objmesh' using per-vertex color
        print('Stage: render SH approximation')
        rshwrapper = rsh_wrapper(self.H, self.W, objmesh)
        for id in tqdm(list_ids):
            c2w = self.list_c2ws[id]
            K = self.list_Ks[id]
            approx = np.zeros((self.H*self.W, 3), dtype=np.uint8)
            rshwrapper.render_approx(c2w, K, approx, L)
            self.approx[id] = approx.astype(np.float32) / 255.

        # compute residual
        if self.need_residual:
            print('Stage: compute residuals')
            for id in tqdm(list_ids):
                self.labels[id] = self.labels[id] - self.approx[id]
            self.residuals = self.labels
        else:
            print('Pass computing residuals')

    def calculate_x_y_z_theta_phi(self, id):
        # Select sample
        depth = self.depths[id]
        c2w = self.list_c2ws[id]
        K = self.list_Ks[id]
        # imageio.imwrite(f'test.png', depth.reshape((self.H, self.W)))

        # constructing xyz coordinates
        ucoords = np.arange(self.W, dtype=np.float32)
        vcoords = np.arange(self.H, dtype=np.float32)
        uvd = np.concatenate([np.stack(np.meshgrid(ucoords, vcoords), -1).reshape(-1, 2), depth], axis=1)
        mask = uvd[:, 2] != 0.
        uvd = uvd[mask]
        uvd[:,:2] = uvd[:,:2] * uvd[:,2:3]
        x_train = np.matmul(uvd, np.matmul(c2w[:3,:3], np.linalg.inv(K)).T) + c2w[:3, 3].T

        # Now x_train is a Mx3 matrix containing position coordinates, where M
        # denote # of non-zero pixels. Next we append directions to x_train
        dirs = x_train - c2w[:3, 3].T
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        x_train = np.hstack([x_train, dirs])

        # Now we have x_train a Mx5 matrix containing position and rotation 
        # coordinates. The label can be easily obtained by taking the mask
        # of label, resulting a Mx3 matrix.
        return x_train, mask

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        # Select sample
        id = self.list_ids[index]
        mask = self.masks[id]

        if self.need_residual:
            return (self.x_trains[id].astype(np.float32), 
                self.residuals[id][mask].astype(np.float32), 
                self.masks[id], 
                self.approx[id][mask].astype(np.float32))
        else:
            return (self.x_trains[id].astype(np.float32), 
                np.zeros_like(self.approx[id][mask], dtype=np.float32), 
                self.masks[id], 
                self.approx[id][mask].astype(np.float32))

# # How to use this script
# # ------
# data_dir = 'data/leather'
# obj_path = os.path.join(data_dir, 'leather-sh.obj')
# L = 0
# dsize = [1024, 1024]

# partition = {'train': [f'./train/r_{i}' for i in range(800)], 
#              'test': [f'./test/r_{i}' for i in range(200)]}
# params = {'shuffle': False,
#           'num_workers': 1}

# training_set = ApproxResSurfaceDataset('blender', data_dir, obj_path, \
#         partition['train'], 'transforms_train.json', need_residual=True, L=L)
# training_generator = torch.utils.data.DataLoader(training_set, **params)

# for i, (x, residual, mask, approx) in enumerate(tqdm(training_generator)):
#     I = np.zeros((dsize[0]*dsize[1], 3))
#     I[mask[0]] = np.abs(residual[0]) * 255
#     I = I.astype(np.uint8).reshape((dsize[0], dsize[1], 3))
#     I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
#     assert(cv2.imwrite(f'test/{i}.png', I))

# test_set = ApproxResSurfaceDataset('blender', data_dir, obj_path, \
#         partition['test'], 'transforms_test.json', need_residual=True, dsize=dsize, L=L)
# test_generator = torch.utils.data.DataLoader(test_set, **params)
# psnrs = np.zeros((len(partition['test']),))

# for i, (x, residual, mask, approx) in enumerate(tqdm(test_generator)):
#     I = np.zeros((dsize[0]*dsize[1], 3))
#     I[mask[0]] = np.abs(approx[0]) * 255
#     I = I.astype(np.uint8).reshape((dsize[0], dsize[1], 3))
#     I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
#     assert(cv2.imwrite(f'test/test_{i}.png', I))
    
#     mse = .5 * .5 * torch.mean(residual ** 2)
#     psnrs[i] = -10. * torch.log10(2.*mse)
