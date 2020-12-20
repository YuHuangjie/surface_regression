import os

#!/bin/bash

gffm_pos,gffm_dir = 1000, 6
map_sizes = [4096]
for map_size in map_sizes:
    cmd = f'python residual_regression.py --model gffm --config ./configs/materials_random.txt --exp materials_random --gffm_pos {gffm_pos} \
    --gffm_dir {gffm_dir} --gffm_map_size {map_size} --use_batch --expname gffm-use-batch'
    print(cmd)
    os.system(cmd)


# map_sizes = [128,256,512,1024,2048,4096]
# for map_size in map_sizes:
#     cmd = f'python residual_regression.py --model ffm --config ./configs/materials_random.txt --exp materials_random --expname ffm --ffm_map_size {map_size}'
#     print(cmd)
#     os.system(cmd)

# gffm_poss, gffm_dirs = [3, 286, 572, 857, 1143, 1428, 1714, 2000], [161, 49, 264, 4, 247, 20,  176, 135]
# map_size = 4096
# for gffm_pos,gffm_dir in zip(gffm_poss,gffm_dirs):
#     cmd = f'python residual_regression.py --model gffm --config ./configs/materials_random.txt --exp materials_random --gffm_pos {gffm_pos} --gffm_dir {gffm_dir} --gffm_map_size {map_size}'
#     print(cmd)
#     os.system(cmd)

# gffm_pos,gffm_dir = 1000, 6
# map_size = 4096
# degrees = [0.5,0.6,0.7,0.8,0.9]
# for degree in degrees:
#     cmd = f'python residual_regression.py --model gffm --config ./configs/materials_random.txt --exp materials_random --gffm_pos {gffm_pos} --expname pfc-{degree} --degree {degree} --gffm_dir {gffm_dir} --gffm_map_size {map_size}'
#     os.system(cmd)


# gffm_pos,gffm_dir = 1000, 6
# map_size = 4096
# degrees = [0.5,0.72]
# for degree in degrees:
#     cmd = f'python residual_regression.py --model gffm --config ./configs/materials_random.txt --exp materials_random --gffm_pos {gffm_pos} --expname pfc-{degree}-first-layer-only --degree {degree} --gffm_dir {gffm_dir} --gffm_map_size {map_size}'
#     os.system(cmd)