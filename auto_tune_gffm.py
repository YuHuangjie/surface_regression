import os
import matplotlib.pyplot as plt
import numpy as np

# tune gffm different hyperparameters
pos_size = 2**np.arange(4, 11, 0.25)
dir_size = np.array([3,6,9,12,15])

for d in dir_size:
        for p in pos_size:
                command = f'python residual_regression.py --config configs/materials.txt '+\
                        f'--train_images=200 --num_epochs=100 --logdir=logs/materials-tune-gffm/pos_{p}_dir_{d} --model gffm --gffm_pos {p} --gffm_dir {d}'
                print(command)
                os.system(command)

'''
Draw error curve
'''
for d in dir_size:
        mean = np.zeros((len(pos_size,)))
        for i, p in enumerate(pos_size):
                result_path = f'logs/materials-tune-gffm/pos_{p}_dir_{d}/gffm-L-1/result/test_psnr.npy'
                psnr = np.load(result_path)
                mse = 10**(-psnr/10)
                mean[i] = mse.mean()
                mean[i] = 10*np.log10(1./mean[i])
                
        plt.plot(np.array(pos_size), mean, label=r'$\theta_{\omega}='+f'{d}'+r'$')

ffm_result = f'logs/materials-tune-ffm/2.8284271247461903/ffm-L-1/result/test_psnr.npy'
psnr = np.load(ffm_result)
mse = 10**(-psnr/10)
psnr = 10*np.log10(1./mse.mean())
plt.axhline(psnr, color='black', linestyle='--', label='best FFM')

plt.xscale('log', basex=2)
plt.grid(True, which='major', alpha=.3)
plt.xlabel(r'positional scale $\theta_{\mu}$')
plt.xlim((pos_size[0], pos_size[-1]))
plt.ylabel('Mean PSNR')
plt.legend()
plt.savefig('fig_gffm_sweep.png')
plt.show()
