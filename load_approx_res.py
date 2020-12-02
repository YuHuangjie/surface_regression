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
    def __init__(self, dataset, datadir, objmesh, list_ids, profile_file, L=-1):
        super(ApproxResSurfaceDataset).__init__()

        self.list_ids = list_ids

        self.x_trains = {}
        self.masks = {}
        self.labels = {}
        self.residuals = {}
        self.approx = {}

        if dataset == 'dslf':
            self.list_c2ws, self.list_Ks, self.list_images = load_dslf_config(
                datadir, profile_file)
        elif dataset == 'blender':
            self.list_c2ws, self.list_Ks, self.list_images = load_blender_config(
                datadir, profile_file)
        else:
            raise RuntimeError(f'Unknown dataset type {dataset}')

        # read ground-truths
        print('Stage: loading images')
        for id in tqdm(list_ids):
            imagepath = self.list_images[id]
            I = imageio.imread(imagepath, pilmode='RGB')
            I = (np.array(I) / 255.).astype(np.float32)
            self.H, self.W = I.shape[:2]
            self.labels[id] = I.reshape((-1, 3))

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
        if L >= 0:
            rshwrapper = rsh_wrapper(self.H, self.W, objmesh)
            for id in tqdm(list_ids):
                c2w = self.list_c2ws[id]
                K = self.list_Ks[id]
                approx = np.zeros((self.H*self.W, 3), dtype=np.uint8)
                rshwrapper.render_approx(c2w, K, approx, L)
                self.approx[id] = approx.astype(np.float32) / 255.
        else:
            for id in list_ids:
                self.approx[id] = np.zeros((self.H*self.W,3),dtype=np.float32)

        # compute residual
        print('Stage: compute residuals')
        for id in list_ids:
            self.labels[id] = self.labels[id] - self.approx[id]
        self.residuals = self.labels

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

        return (self.x_trains[id].astype(np.float32), 
            self.residuals[id][mask].astype(np.float32), 
            self.masks[id], 
            self.approx[id][mask].astype(np.float32))
