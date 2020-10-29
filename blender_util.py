import numpy as np
import os
import json
import imageio

def load_blender_config(datadir, profile_file):
    list_poses = {}
    list_Ks = {}
    list_images = {}
    probed = False

    with open(os.path.join(datadir, profile_file), 'r') as fp:
        meta = json.load(fp)
        frames = meta['frames']
        camera_angle_x = meta['camera_angle_x']

    for frame in frames:
        file_path = frame['file_path']
        rotation = frame['rotation']
        tm = np.array(frame['transform_matrix'])

        # Note that *.obj exported from blender has a different CS
        # than in blender. The transformation matrix is described
        # as:
        #       [1, 0, 0]
        #   M = [0, 0, 1]
        #       [0,-1, 0]
        M = np.array([[1,0,0], [0,0,1], [0,-1,0]])
        tm[:3,:3] = np.matmul(M, tm[:3,:3])
        tm[:3, 3] = np.matmul(M, tm[:3, 3])
        # Also note that the depth rendering engine requires that 
        # y axis points downward instead of upward :)
        tm[:3, 1] = -tm[:3, 1]
        tm[:3, 2] = -tm[:3, 2]

        id = file_path
        list_poses[id] = tm
        list_images[id] = os.path.join(datadir, f'{id}.0000.png')

        if not probed:
            probe_i = imageio.imread(list_images[id])
            H, W = probe_i.shape[:2]
            probed = True
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        list_Ks[id] = np.array([[focal, 0, W/2], 
                               [0, focal, H/2], 
                               [0, 0, 1]])

    return (list_poses, list_Ks, list_images)
        