import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# run ffm with different mapping scales
map_size = 2**np.arange(0, 4, 0.5)

for m in map_size:
        command = f'python residual_regression.py --config configs/materials.txt '+\
                f'--train_images=200 --num_epochs=100 --logdir=logs/materials-tune-ffm/{m} --model ffm --ffm_map_scale {m}'
        print(command)
        os.system(command)

# run gffm with different pairs of hyperparameters
pos_size = 2**np.arange(4, 11, 0.25)
dir_size = np.array([3,6,9,12,15])

for d in dir_size:
        for p in pos_size:
                command = f'python residual_regression.py --config configs/materials.txt '+\
                        f'--train_images=200 --num_epochs=100 --logdir=logs/materials-tune-gffm/pos_{p}_dir_{d} --model gffm --gffm_pos {p} --gffm_dir {d}'
                print(command)
                os.system(command)

'''
Make figure
'''
params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize': 14,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
matplotlib.rcParams.update(params)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
ax = plt.gca()
mean = np.zeros((len(map_size,)))
best_ffm_psnr = 0
for i, length in enumerate(map_size):
        result_path = f'logs/materials-tune-ffm/{length}/ffm-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
        if mean[i] > best_ffm_psnr:
                best_ffm_psnr = mean[i]
ax.plot(np.array(map_size), mean)
ax.set_xlim((map_size[0], map_size[-1]))
ax.set_xlabel('(a) FFM mapping scale')
ax.grid(True, which='major', alpha=.3)
ax.set_xscale('log', basex=2)
ax.set_ylabel('Mean PSNR')

plt.subplot(1,2,2)
ax = plt.gca()
for d in dir_size:
        mean = np.zeros((len(pos_size,)))
        for i, p in enumerate(pos_size):
                result_path = f'logs/materials-tune-gffm/pos_{p}_dir_{d}/gffm-L-1/result/test_psnr.npy'
                psnr = np.load(result_path)
                mse = 10**(-psnr/10)
                mean[i] = mse.mean()
                mean[i] = 10*np.log10(1./mean[i])
                
        ax.plot(np.array(pos_size), mean, label=r'$\theta_{\omega}='+f'{d}'+r'$')

ax.axhline(best_ffm_psnr, color='black', linestyle='--', label='best FFM')
ax.set_xlabel(r'(b) positional scale $\theta_{\mu}$')
ax.set_xlim((pos_size[0], pos_size[-1]))
ax.grid(True, which='major', alpha=.3)
ax.set_xscale('log', basex=2)
ax.set_ylabel('Mean PSNR')

plt.legend(loc='center left', bbox_to_anchor=(1,.5), handlelength=1)
plt.tight_layout()
plt.savefig('fig_gffm_sweep.png')
plt.show()
