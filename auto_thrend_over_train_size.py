import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# batch with different size of training images
training_size = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

for length in training_size:
        '''
        RELU
        '''
        command = f'python residual_regression.py --config configs/materials.txt '+\
                f'--train_images={length} --num_epochs=100 --logdir=logs/materials-vary-train_images/{length} --model relu'
        print(command)
        os.system(command)

        '''
        FFM
        '''
        command = f'python residual_regression.py --config configs/materials.txt '+\
                f'--train_images={length} --num_epochs=100 --logdir=logs/materials-vary-train_images/{length} --model ffm'
        print(command)
        os.system(command)

        '''
        GFFM
        '''
        command = f'python residual_regression.py --config configs/materials.txt '+\
                f'--train_images={length} --num_epochs=100 --logdir=logs/materials-vary-train_images/{length} --model gffm'
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

mean = np.zeros((len(training_size,)))
for i, length in enumerate(training_size):
        result_path = f'logs/materials-vary-train_images/{length}/relu-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
plt.plot(np.array(training_size), mean, label='Relu')

mean = np.zeros((len(training_size,)))
for i, length in enumerate(training_size):
        result_path = f'logs/materials-vary-train_images/{length}/ffm-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
plt.plot(np.array(training_size), mean, label='FFM')

mean = np.zeros((len(training_size,)))
for i, length in enumerate(training_size):
        result_path = f'logs/materials-vary-train_images/{length}/gffm-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
plt.plot(np.array(training_size), mean, label='GFFM')

plt.grid(True, which='major', alpha=.3)
plt.xlim((training_size[0], training_size[-1]))
plt.xlabel('number of training images')
plt.ylabel('Mean PSNR')
plt.legend()
plt.savefig('fig_trend_over_training_size.png')
plt.show()
