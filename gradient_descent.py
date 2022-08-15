import numpy as np
import matplotlib.pyplot as plt

weights = np.array([99.79, 117.93, 95.25, 100.70, 108.86, 87.54])
heights = np.array([198.12, 215.90, 205.74, 203.20, 205.74, 190.05])
m = 0.5; c = 150
y = m*weights + c

# NOTE font and its size
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# plt.rc('font', **font)
SMALL_SIZE = 22
MEDIUM_SIZE = 10
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.figure(figsize=(15, 15))
plt.scatter(weights, heights, s=100, c='orange', edgecolors='black', alpha=1.0, marker='o')
plt.plot(weights, y, 'r')
plt.title('ccc')
plt.xlabel('xxx')
plt.ylabel('yyy')
# plt.savefig('image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()