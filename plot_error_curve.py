import matplotlib.pyplot as plt
import numpy as np

training_size = [100, 200, 300, 400, 500, 600, 700, 800]

cell_text = []

'''
Draw error curve
'''
mean = np.zeros((len(training_size,)))
stddev = np.zeros((len(training_size,)))
for i, length in enumerate(training_size):
        result_path = f'logs/material-vary-length/relu-{length}/relu-L-1/result/test_psnr.txt'
        psnr = np.loadtxt(result_path, comments=',')
        mean[i] = psnr.mean()
        stddev[i] = np.std(psnr)
plt.errorbar(np.array(training_size), mean, yerr=stddev, fmt='-o', label='relu')
cell_text.append([f'{m:.3f}' for m in mean])

mean = np.zeros((len(training_size,)))
stddev = np.zeros((len(training_size,)))
for i, length in enumerate(training_size):
        result_path = f'logs/material-vary-length/ffm-{length}/ffm-L-1/result/test_psnr.txt'
        psnr = np.loadtxt(result_path, comments=',')
        mean[i] = psnr.mean()
        stddev[i] = np.std(psnr)
plt.errorbar(np.array(training_size), mean, yerr=stddev, fmt='-o', label='ffm')
cell_text.append([f'{m:.3f}' for m in mean])

mean = np.zeros((len(training_size,)))
stddev = np.zeros((len(training_size,)))
for i, length in enumerate(training_size):
        result_path = f'logs/material-vary-length/gffm-{length}/gffm-L-1/result/test_psnr.txt'
        psnr = np.loadtxt(result_path, comments=',')
        mean[i] = psnr.mean()
        stddev[i] = np.std(psnr)
plt.errorbar(np.array(training_size), mean, yerr=stddev, fmt='-o', label='gffm')
cell_text.append([f'{m:.3f}' for m in mean])

plt.legend()
plt.title('mean-stddev of 200 tests')
plt.savefig('error-bar.png')
plt.clf()

'''
table
'''
rows = ['relu', 'ffm', 'gffm']
columns = [f'{ts}' for ts in training_size]

plt.figure(figsize=(10, 3))
plt.table(cellText=cell_text,
        rowLabels=rows,
        colLabels=columns,
        loc='center')
# Adjust layout to make room for the table:
plt.axis('off')
plt.title('mean PSNR of 200 tests')
plt.savefig('table.png')
