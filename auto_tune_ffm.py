import os
import matplotlib.pyplot as plt
import numpy as np

# run ffm with different mapping scales
map_size = 2**np.arange(0, 4, 0.5)

for m in map_size:
        command = f'python residual_regression.py --config configs/materials.txt '+\
                f'--train_images=200 --num_epochs=100 --logdir=logs/material-tune-ffm/{m} --model ffm --ffm_map_scale {m}'
        print(command)
        os.system(command)

'''
Draw error curve
'''
mean = np.zeros((len(map_size,)))
for i, length in enumerate(map_size):
        result_path = f'logs/material-tune-ffm/{length}/ffm-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
        print(f'mapping scale: {length:.02f}, mean psnr: {mean[i]:.02f}')

plt.plot(np.array(map_size), mean)
plt.grid(True, which='major', alpha=.3)
plt.xlim((map_size[0], map_size[-1]))
plt.xscale('log', basex=2)
plt.xlabel('FFM mapping scale')
plt.ylabel('Mean PSNR')
plt.savefig('fig_ffm_sweep.png')
plt.show()
