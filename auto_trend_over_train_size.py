import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

data_id = 'materials'

'''
Batch training with small number of images
'''
training_size = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600]
small_training_size = [10,20,30,40,50,60,70,80,90,100]
large_training_size = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600]

for length in large_training_size:
        '''
        RELU
        '''
        command = f'python residual_regression.py --config configs/{data_id}.txt '+\
                f'--train_images={length} --num_epochs=100 --logdir=logs/{data_id}-vary-train_images/{length} --model relu'
        print(command)
        os.system(command)

        '''
        FFM
        '''
        command = f'python residual_regression.py --config configs/{data_id}.txt '+\
                f'--train_images={length} --num_epochs=100 --logdir=logs/{data_id}-vary-train_images/{length} --model ffm'
        print(command)
        os.system(command)

        '''
        GFFM
        '''
        command = f'python residual_regression.py --config configs/{data_id}.txt '+\
                f'--train_images={length} --num_epochs=100 --logdir=logs/{data_id}-vary-train_images/{length} --model gffm'
        print(command)
        os.system(command)

'''
Make figure
'''
params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize': 13,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
matplotlib.rcParams.update(params)


## plot performance trained with a small number of images
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
ax = plt.gca()

mean = np.zeros((len(small_training_size,)))
for i, length in enumerate(small_training_size):
        result_path = f'logs/{data_id}-vary-train_images/{length}/relu-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
ax.plot(np.array(small_training_size), mean, label='Relu', c='blue')

for i, length in enumerate(small_training_size):
        result_path = f'logs/{data_id}-vary-train_images/{length}/ffm-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
ax.plot(np.array(small_training_size), mean, label='FFM', c='red')

for i, length in enumerate(small_training_size):
        result_path = f'logs/{data_id}-vary-train_images/{length}/gffm-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
ax.plot(np.array(small_training_size), mean, label='GFFM', c='green')

for i, length in enumerate(small_training_size):
        result_path = f'../nngp/exp/{data_id}-{length}/test_psnr.npy'
        psnr = np.load(result_path)
        mean[i] = psnr.mean()
ax.plot(np.array(small_training_size), mean, label='NNGP', color='black')

ax.grid(True, which='major', alpha=.3)
ax.set_xlim((small_training_size[0], small_training_size[-1]))
ax.set_xlabel('Training size')
ax.set_title('(a) Train with less images',  y=-0.3)
ax.set_ylabel('Mean PSNR')
ax.legend(loc='lower right')


## plot performance trained with a large number of images
plt.subplot(1,2,2)
ax = plt.gca()

mean = np.zeros((len(large_training_size,)))
for i, length in enumerate(large_training_size):
        result_path = f'logs/{data_id}-vary-train_images/{length}/relu-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
ax.plot(np.array(large_training_size), mean, label='Relu', c='blue')

for i, length in enumerate(large_training_size):
        result_path = f'logs/{data_id}-vary-train_images/{length}/ffm-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
ax.plot(np.array(large_training_size), mean, label='FFM', c='red')

for i, length in enumerate(large_training_size):
        result_path = f'logs/{data_id}-vary-train_images/{length}/gffm-L-1/result/test_psnr.npy'
        psnr = np.load(result_path)
        mse = 10**(-psnr/10)
        mean[i] = mse.mean()
        mean[i] = 10*np.log10(1./mean[i])
ax.plot(np.array(large_training_size), mean, label='GFFM', c='green')

ax.grid(True, which='major', alpha=.3)
ax.set_xlim((large_training_size[0], large_training_size[-1]))
ax.set_xlabel('Training size')
ax.set_ylabel('Mean PSNR')
ax.set_title('(b) Train with more images', y=-.3)
ax.legend(loc='lower right')


## Save figure
plt.tight_layout()
plt.savefig('fig_trend_over_training_size.png')
plt.show()
