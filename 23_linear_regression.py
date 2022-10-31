import numpy as np
import matplotlib.pyplot as plt

# NOTE input data
weights = np.array([99.79, 117.93, 95.25, 100.70, 108.86, 87.54])
heights = np.array([198.12, 215.90, 205.74, 203.20, 205.74, 190.05])
m = 0.5; c = 150
y = m*weights + c

# NOTE plot
plt.figure(figsize=(20, 15))
plt.rcParams.update({'font.size': 22})
plt.scatter(weights, heights, s=300, c='orange', edgecolors='black', alpha=1.0, marker='o')
plt.plot(weights, y, 'r')
plt.title('Linear Regression', fontweight='bold')
plt.xlabel('weight', fontweight='bold')
plt.ylabel('height', fontweight='bold')
# plt.savefig('image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()