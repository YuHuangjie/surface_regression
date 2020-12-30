import numpy as np
import os
import json

def load_dslf_config(datadir, profile_file):
    list_poses = {}
    list_Ks = {}
    list_images = {}

    with open(os.path.join(datadir, profile_file), 'r') as fp:
        meta = json.load(fp)
        frames = meta['frames']

    for frame in frames:
        tag = frame['tag']
        file_path = frame['file_path']
        extrinsic = frame['extrinsic']
        intrinsic = frame['intrinsic']

        list_images[tag] = os.path.join(datadir, file_path)

        # parse intrinsic
        M = np.zeros((9, 1))
        intrinsic = intrinsic.split()
        for i in range(9):
            M[i] = float(intrinsic[i])
        M = np.reshape(M, (3,3))
        list_Ks[tag] = M.astype(np.float32)

        # parse camera poses
        M = np.zeros((16, 1))
        extrinsic = extrinsic.split()
        for i in range(16):
            M[i] = float(extrinsic[i])
        M = np.reshape(M, (4,4))
        list_poses[tag] = M.astype(np.float32)

    return (list_poses, list_Ks, list_images)
