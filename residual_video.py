import torch
import numpy as np
import os, imageio

from load_vertex_color_residual import VertexColorResidualSurfaceDataset
from model import *

device = torch.device('cuda', 1)

data_dir = 'data/tangsancai'
obj_path = 'data/tangsancai/obj/mesh_vc.obj'
dsize = [400, 640]
exp_dir = 'exp/tangsancai/gauss_0.2'

# Set up testing data generator
partition = {'video': [f'{i}' for i in range(825, 1127)]}
params = {'shuffle': False,
          'num_workers': 1,}

video_set = VertexColorResidualSurfaceDataset(data_dir, obj_path, 
        partition['video'], 'profile.txt', need_residual=False, dsize=dsize)
video_generator = torch.utils.data.DataLoader(video_set, **params)

# load saved model
saved_model_dict = torch.load(os.path.join(exp_dir, 'model.pt'))
D = saved_model_dict['D']
W = saved_model_dict['W']
B = saved_model_dict['B']
B_view = saved_model_dict['B_view']
model = make_network(D, W, B, B_view).to(device)
model.load_state_dict(saved_model_dict, strict=False)
model.eval()

# create output directory
output_dir = f'{exp_dir}/video'
os.makedirs(output_dir, exist_ok=True)

# make predictions
with torch.no_grad():
        for i, (x, mask, approx) in enumerate(video_generator):
                x = x.to(device)
                y = model_pred(model, x)
                img = torch.zeros((dsize[0]*dsize[1], 3))
                img[mask[0]] = y.cpu() + approx[0]
                img = np.clip(img.numpy() * 255., 0, 255).astype(np.uint8)
                img = img.reshape(dsize + [3])

                imageio.imwrite(f'{output_dir}/{i}.png', img)