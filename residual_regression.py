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
import configargparse

from rd_wrapper import rd_wrapper
from load_approx_res import ApproxResSurfaceDataset as Dataset
from model import *
from train import *
from sample.sample import slf_sample, slf_sample_init

p = configargparse.ArgumentParser()

p.add_argument('--config', is_config_file=True, help='config file path')
p.add_argument('--logdir', type=str, required=False, default='./logs/default', help='root for logging')
p.add_argument('--test_only', action='store_true', help='test only')
p.add_argument('--restart', action='store_true', help='do not reload from checkpoints')
p.add_argument('--datatype', type=str, default='blender',help='data loader type (blender or dslf)')
p.add_argument('--exp', type=str, required=True, help='identifier of training data (e.g. lucy)')
p.add_argument('--sh_level', type=int, default=3, help='order of SH basis (0-3)')   
p.add_argument('--batch_rays', type=int, default=100000, help='number of rays per batch')   

# General training options
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for.')
p.add_argument('--net_depth', type=int, default=8)
p.add_argument('--net_width', type=int, default=256)

p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Epoch interval until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Step interval until loss is printed.')
p.add_argument('--train_images', type=int, default=100,
                help='number of training images')
p.add_argument('--test_images', type=int, default=200,
                help='number of testing images')

p.add_argument('--model', type=str, action='append', required=True,
               help='Options available are "relu", "ffm", "gffm"')
p.add_argument('--ffm_map_size', type=int, default=1024,
               help='mapping dimension of ffm')
p.add_argument('--ffm_map_scale', type=float, default=10,
               help='Gaussian mapping scale of positional input')
p.add_argument('--gffm_map_size', type=int, default=4096,
               help='mapping dimension of gffm')
p.add_argument('--gffm_pos', type=float, default=1000,
               help='mapping dimension of gffm')
p.add_argument('--gffm_dir', type=int, default=6,
               help='mapping dimension of gffm')
args = p.parse_args()

datatype = args.datatype
data_dir = f'data/{args.exp}'

# Set up training/testing data
if datatype == 'blender':
    train_part = [f'./train/r_{i}' for i in range(args.train_images)]
    test_part =   [f'./test/r_{i}' for i in range(args.test_images)]
    train_params = {'shuffle': True, 'num_workers': 24, 'pin_memory': True, 'persistent_workers': True}
    test_params = {'shuffle': False, 'num_workers': 0, 'pin_memory': False,}
    obj_path = f'{data_dir}/{args.exp}-sh.obj'

    if not args.test_only:
        train_set = Dataset(datatype, data_dir, obj_path, train_part, 
                    'transforms_train.json', L=args.sh_level, train=True, n_rays=args.batch_rays)
        train_dataloader = torch.utils.data.DataLoader(train_set, **train_params)

    test_set = Dataset(datatype, data_dir, obj_path, test_part, 
                    'transforms_test.json', L=args.sh_level, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_set, **test_params)
else:
    raise NotImplementedError

# train each model configuration
for mt in args.model:
    tqdm.write(f'Running at {args.exp} / {mt}')

    # Load checkpoints
    logdir = os.path.join(args.logdir, f'{mt}-L{args.sh_level}')
    global_step = 0
    model_params = None
    state_dict = None
    if os.path.exists(os.path.join(logdir, 'checkpoints')):
        ckpts = [os.path.join(logdir, 'checkpoints', f) for f in sorted(os.listdir(os.path.join(logdir, 'checkpoints'))) if 'pt' in f]
        if len(ckpts) > 0 and not args.restart:
            ckpt_path = ckpts[-1]
            tqdm.write(f'Reloading from {ckpt_path}')
            ckpt = torch.load(ckpt_path)
            global_step = ckpt['global_step']
            model_params = ckpt['params']
            state_dict = ckpt['model']

    # network architecture
    network_size = (args.net_depth, args.net_width)

    if mt == 'relu':
        model = make_relu_network(*network_size)
    elif mt == 'ffm':
        if model_params is None:
            B = torch.normal(0, 1, size=(args.ffm_map_size, 6)) * args.ffm_map_scale
        else:
            B = model_params
        model = make_ffm_network(*network_size, B)
        model_params = B
    elif mt == 'gffm':
        if model_params is None:
            tqdm.write(f'sampling SLF kernel with params ({args.gffm_pos}, {args.gffm_dir}), might take a while')
            integrand_lib = slf_sample_init('sample/integrands.so')
            W = slf_sample(integrand_lib, args.gffm_pos, args.gffm_dir, N=args.gffm_map_size)
            b = np.random.uniform(0, 2*np.pi, size=(1, args.gffm_map_size))
            tqdm.write(f'finish sampling')
        else:
            (W, b) = model_params
        model = make_rff_network(*network_size, W, b)
        model_params = (W, b)
    else:
        raise NotImplementedError

    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.cuda()

    # training
    if not args.test_only:
        train(model, train_dataloader, args.lr, epochs=args.num_epochs, logdir=logdir,
            epochs_til_checkpoint=args.epochs_til_ckpt, steps_til_summary=args.steps_til_summary,
            val_dataloader=None, global_step=global_step, model_params=model_params)

    # make full testing
    tqdm.write("Running full validation set...")
    output_dir = os.path.join(logdir, 'result')
    os.makedirs(output_dir, exist_ok=True)
    psnr = []
    dsize = (test_set.H, test_set.W)

    with torch.no_grad():
        for i, (x, residual, mask, approx) in enumerate(test_dataloader):
            x, residual = x.cuda(), residual.cuda()
            y = model_pred(model, x)
            img = torch.zeros((dsize[0]*dsize[1], 3))
            img[mask[0]] = y.cpu() + approx[0].cpu()
            img = np.clip(img.numpy() * 255., 0, 255).astype(np.uint8)
            img = img.reshape(dsize + (3,))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assert(cv2.imwrite(f'{output_dir}/{i}.png', img))
            psnr.append(model_psnr(model_loss(y, residual)).item())

    # save test psnrs
    np.savetxt(f'{output_dir}/test_psnr.txt', psnr, newline=',\n')
    np.save(f'{output_dir}/test_psnr.npy', psnr)
