import numpy as np
import os
import imageio
from math import pi, sin, cos, sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt

from rd_wrapper import rd_wrapper
from dslf_util import load_dslf_config
from blender_util import load_blender_config

def _create_line_generator(file_name):
        """
        Creates a generator function yielding lines in the file
        Should only yield non-empty lines
        """
        file = open(str(file_name), mode='r')

        for line in file:
            yield line

        file.close()

def SH(l, m):
    '''
    return a Spherical Harmonic basis function
    l is the band, range [0...N]
    m in the range [-l, l]
    '''
    # Y_0
    if l == 0:
        return lambda x, y, z: sqrt(1/pi)/2
    # Y_1
    elif l == 1 and m == -1:
        return lambda x, y, z: sqrt(3/pi/4)*y/sqrt(x*x+y*y+z*z)
    elif l == 1 and m == 0:
        return lambda x, y, z: sqrt(3/pi/4)*z/sqrt(x*x+y*y+z*z)
    elif l == 1 and m == 1:
        return lambda x, y, z: sqrt(3/pi/4)*x/sqrt(x*x+y*y+z*z)
    # Y_2
    elif l == 2 and m == -2:
        return lambda x, y, z: sqrt(15/pi/4)*x*y/(x*x+y*y+z*z)
    elif l == 2 and m == -1:
        return lambda x, y, z: sqrt(15/pi/4)*y*z/(x*x+y*y+z*z)
    elif l == 2 and m == 0:
        return lambda x, y, z: sqrt(5/pi/16)*(-x*x-y*y+2*z*z)/(x*x+y*y+z*z)
    elif l == 2 and m == 1:
        return lambda x, y, z: sqrt(15/pi/4)*z*x/(x*x+y*y+z*z)
    elif l == 2 and m == 2:
        return lambda x, y, z: sqrt(15/pi/16)*(x*x-y*y)/(x*x+y*y+z*z)
    # Y_3
    elif l == 3 and m == -3:
        return lambda x, y, z: sqrt(35/pi/32)*(3*x*x-y*y)*y/(x*x+y*y+z*z)/sqrt(x*x+y*y+z*z)
    elif l == 3 and m == -2:
        return lambda x, y, z: sqrt(105/pi/4)*x*y*z/(x*x+y*y+z*z)/sqrt(x*x+y*y+z*z)
    elif l == 3 and m == -1:
        return lambda x, y, z: sqrt(21/pi/32)*y*(4*z*z-x*x-y*y)/(x*x+y*y+z*z)/sqrt(x*x+y*y+z*z)
    elif l == 3 and m == 0:
        return lambda x, y, z: sqrt(7/pi/16)*z*(2*z*z-3*x*x-3*y*y)/(x*x+y*y+z*z)/sqrt(x*x+y*y+z*z)
    elif l == 3 and m == 1:
        return lambda x, y, z: sqrt(21/pi/32)*x*(4*z*z-x*x-y*y)/(x*x+y*y+z*z)/sqrt(x*x+y*y+z*z)
    elif l == 3 and m == 2:
        return lambda x, y, z: sqrt(105/pi/16)*(x*x-y*y)*z/(x*x+y*y+z*z)/sqrt(x*x+y*y+z*z)
    elif l == 3 and m == 3:
        return lambda x, y, z: sqrt(35/pi/32)*(x*x-3*y*y)*x/(x*x+y*y+z*z)/sqrt(x*x+y*y+z*z)
    else:
        raise RuntimeError("NO such implementation")
        

def _sh_approx(input_obj, output_obj, list_ids, list_c2ws, list_Ks, list_gts, EPS, L):
        (H, W) = list_gts[list_ids[0]].shape[0], list_gts[list_ids[0]].shape[1]
        rdwrapper = rd_wrapper(H, W, input_obj)
        depths = {}
        list_w2cs = {}
        count = 0

        # generate a line in input_obj each time
        line_generator = _create_line_generator(input_obj)
        lines = []

        # render depth maps
        print('Stage: loading depths')
        for id in tqdm(list_ids):
            c2w = list_c2ws[id]
            K = list_Ks[id]
            depth = np.zeros((H * W, 1), dtype=np.float32)
            rdwrapper.render_depth(c2w, K, depth)
            depths[id] = depth.reshape((H, W))
            list_w2cs[id] = np.linalg.inv(c2w)

        # sh orders
        sh_funs = [SH(l, m) for l in range(L+1) for m in range(-l, l+1)]

        try:
            while True:
                line = next(line_generator)

                if line[0] != 'v' or line[1] != ' ':
                    lines.append(line)
                    continue

                count+=1
                if count % 100 == 0:
                    print(f'processing {count}', end='\r')

                # vertex Cartesian coordinates
                values = line.split()
                pglb = np.array([float(values[1]), float(values[2]), float(values[3])])
                
                # Will solve for argmin(||Ax-b||^2 )
                A = np.zeros((len(list_ids), len(sh_funs)))
                b = np.zeros((len(list_ids), 3))
                n = 0

                for id in list_ids:
                    w2c = list_w2cs[id]
                    pcam = w2c[:3,:3] @ pglb + w2c[:3, 3]
                    uv = list_Ks[id] @ pcam
                    (u, v) = int(uv[0]/uv[2]), int(uv[1]/uv[2])

                    # texture coordinates sanity check
                    if u < 0 or u >= W or v < 0 or v >= H:
                        continue

                    # depth testing
                    if np.abs(depths[id][v][u] - pcam[2]) > EPS:
                        continue

                    drc = pglb - list_c2ws[id][:3, 3]
                    A[n, :] = [fun(drc[0], drc[1], drc[2]) for fun in sh_funs]
                    b[n, :] = list_gts[id][v][u]
                    n += 1

                # solve argmin(||Ax-b||^2 ) for coefficients order by order
                A = A[:n, :]
                b = b[:n, :]
                coeffs = np.zeros((len(sh_funs), 3))
                if n == 0: sh_L = -1
                elif n <= 4: sh_L = min(0, L)
                elif n <= 9: sh_L = min(1, L)
                elif n <= 16: sh_L = min(2, L)
                else: sh_L = min(3, L)
                
                for l in range(sh_L+1):
                    coeffs[l*l:(l+1)*(l+1), :] = np.linalg.lstsq(A[:,l*l:(l+1)*(l+1)], b)[0]
                    b -= A[:,l*l:(l+1)*(l+1)] @ coeffs[l*l:(l+1)*(l+1), :]

                # linearize co-efficients 
                coeffs = coeffs.reshape([-1])

                line = f'v {pglb[0]} {pglb[1]} {pglb[2]}'
                line += ''.join([f' {coeff}' for coeff in coeffs])
                line += '\n'
                lines.append(line)

        except StopIteration:
            print("Done coloring vertices")

        # output vertex-colored mesh
        with open(output_obj, 'w') as f:
            for line in lines:
                print(line, file = f, end='')

def sh_approx(input_obj, output_obj, dataset, datadir, profile_file, list_ids, EPS, L=1):
    
    if dataset == 'dslf':
        list_poses, list_Ks, list_images = load_dslf_config(datadir, profile_file)
    elif dataset == 'blender':
        list_poses, list_Ks, list_images = load_blender_config(datadir, profile_file)
    else:
        raise RuntimeError(f'Unknown dataset type {dataset}')

    list_gts = {}
    probed = False

    # determine target image size
    probe_image = list_images[list(list_images)[0]]
    probe_image = imageio.imread(probe_image)
    H, W = probe_image.shape[:2]
    probed = True

    # read ground-truths
    print('Stage: loading images')
    for id in tqdm(list_ids):
        imagepath = list_images[id]
        I = imageio.imread(imagepath, pilmode='RGB')
        I = (np.array(I) / 255.).astype(np.float32)
        list_gts[id] = I

    _sh_approx(input_obj, output_obj, list_ids, list_poses, list_Ks, list_gts, EPS, L)


data_dir = 'data/lucy'
input_obj = os.path.join(data_dir, 'lucy.obj')
output_obj = os.path.join(data_dir, 'lucy-sh.obj')
list_ids = [f'./train/r_{i}' for i in range(800)]

sh_approx(input_obj, output_obj, 'blender', data_dir, "transforms_train.json", list_ids, EPS=1e-2, L=3)
