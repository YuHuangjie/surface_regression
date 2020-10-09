from tqdm import tqdm as tqdm
from load_dslf_surface import DslfSurfaceDataset
from rd_wrapper import rd_wrapper
import torch
import os
import numpy as np

# load depth render library
data_dir = 'data/tangsancai'
obj_path = os.path.join(data_dir, 'obj/mesh_regi.obj')
dsize = [250, 400]
rdwrapper = rd_wrapper(*dsize, obj_path)

partition = {'train': [f'{i}' for i in range(369)], 
             'test': [f'{i}' for i in range(369, 370)]}

params = {'shuffle': False,
          'num_workers': 1}

training_set = DslfSurfaceDataset(data_dir, rdwrapper, \
        partition['train'], 'profile.txt', dsize=dsize)
training_generator = torch.utils.data.DataLoader(training_set, **params)

for i, (local_batch, local_labels, mask) in enumerate(tqdm(training_generator)):
        with open(os.path.join('views_np', f'{i}.npy'), 'wb') as f:
                np.save(f, local_batch)
                np.save(f, local_labels)
                np.save(f, mask)
        
test_set = DslfSurfaceDataset(data_dir, rdwrapper, \
        partition['test'], 'profile.txt', dsize=dsize)
test_generator = torch.utils.data.DataLoader(test_set, **params)

for i, (local_batch, local_labels, mask) in enumerate(tqdm(test_generator)):
        with open(os.path.join('views_np', f'test{i}.npy'), 'wb') as f:
                np.save(f, local_batch)
                np.save(f, local_labels)
                np.save(f, mask)