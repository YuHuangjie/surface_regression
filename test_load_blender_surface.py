from tqdm import tqdm as tqdm
from load_blender_surface import BlenderSurfaceDataset
from rd_wrapper import rd_wrapper
import torch
import os
import numpy as np

# load depth render library
data_dir = 'data/hotdog'
obj_path = os.path.join(data_dir, 'hot-dogs.obj')
dsize = [512, 512]
rdwrapper = rd_wrapper(*dsize, obj_path)

partition = {'train': [f'./train/r_{i}' for i in range(100)], 
             'test': [f'./test/r_{i}' for i in range(200)]}

params = {'shuffle': False,
          'num_workers': 1}

training_set = BlenderSurfaceDataset(data_dir, rdwrapper, \
        partition['train'], 'transforms_train.json', dsize=dsize)
training_generator = torch.utils.data.DataLoader(training_set, **params)

for i, (local_batch, local_labels, mask) in enumerate(tqdm(training_generator)):
        with open(os.path.join('views_np', f'{i}.npy'), 'wb') as f:
                np.save(f, local_batch)
                np.save(f, local_labels)
        
test_set = BlenderSurfaceDataset(data_dir, rdwrapper, \
        partition['test'], 'transforms_test.json', dsize=dsize)
test_generator = torch.utils.data.DataLoader(test_set, **params)

for i, (local_batch, local_labels, mask) in enumerate(tqdm(test_generator)):
        with open(os.path.join('views_np', f'test{i}.npy'), 'wb') as f:
                np.save(f, local_batch)
                np.save(f, local_labels)
                np.save(f, mask)