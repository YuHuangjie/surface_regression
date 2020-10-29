import numpy as np
import os

def load_dslf_config(datadir, profile_file):
    extrinsic_file = None
    intrinsic_file = None
    image_file = None
    list_poses = {}
    list_Ks = {}
    list_images = {}

    with open(os.path.join(datadir, profile_file), 'r') as fp:
        for line in fp:
            line = line.strip()
            if line == '':
                continue
            parts = line.split(':')
            if parts[0] == 'camera_pose':
                extrinsic_file = parts[1].strip()
            elif parts[0] == 'camera_intrinsic':
                intrinsic_file = parts[1].strip()
            elif parts[0] == 'image_list':
                image_file = parts[1].strip()

    with open(os.path.join(datadir, image_file), 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            file_name = parts[1].strip()

            # check image existance
            file_name = os.path.join(datadir, file_name)
            if not os.path.exists(file_name):
                raise Exception(f'{file_name} is not found')

            list_images[parts[0]] = file_name
    
    with open(os.path.join(datadir, extrinsic_file), 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split(maxsplit=1)
            extrinsic = parts[1].strip()

            # parse camera poses
            M = np.zeros((16, 1))
            extrinsic = extrinsic.split()
            for i in range(16):
                M[i] = float(extrinsic[i])
            M = np.reshape(M, (4,4))
            M[:, 1] = -M[:, 1]  # y-axis differs
            M[:, 2] = -M[:, 2]  # so does z-axis
            list_poses[parts[0]] = M.astype(np.float32)

    with open(os.path.join(datadir, intrinsic_file), 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split(maxsplit=1)
            intrinsic = parts[1].strip()

            # parse intrinsic
            M = np.zeros((9, 1))
            intrinsic = intrinsic.split()
            for i in range(9):
                M[i] = float(intrinsic[i])
            M = np.reshape(M, (3,3))
            list_Ks[parts[0]] = M.astype(np.float32)

    return (list_poses, list_Ks, list_images)
