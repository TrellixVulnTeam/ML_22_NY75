import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

'''
step 1: input data
'''

weights = np.array([99.79, 117.93, 95.25, 100.70, 108.86, 87.54]) # axis-x
heights = np.array([198.12, 215.90, 205.74, 203.20, 205.74, 190.05]) # axis-y

'''
step 2: predefine variables
'''

size = 6
cost_function = np.zeros(shape=(size, size), dtype=np.float64)
beg_m = 0.49; end_m = 0.64
# NOTE fix intercept
c = 146
m = np.linspace(beg_m, end_m, size) # axis x (slope)

'''
step 2: plot iterative MSEs
'''

# plt.figure(figsize=(10, 8))
# plt.rcParams.update({'font.size': 12})
# plt.scatter(weights, heights, s=300, c='orange', edgecolors='black', alpha=1.0, marker='o')
# color = ['red','orange','black','green','blue','purple']
# for index, i in enumerate(m):
# 	y = F.linear_equation(i, weights, c)
# 	error = F.MSE(heights, y)
# 	plt.plot(weights, y, c=color[index],
#              label='MSE: ' + str('%.2f' % error) + ' slope: ' + str('%.2f' % i))
# plt.title('Iterative MSEs', fontweight='bold')
# plt.xlabel('weight', fontweight='bold')
# plt.ylabel('height', fontweight='bold')
# plt.legend()
# # plt.savefig('../drawing/image_out/' + 'i_mses' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()

'''
step 3: plot discreated cost function
'''

cf_slope = np.zeros(shape=(len(m), ), dtype=np.float64)
for index, i in enumerate(m):
	y = F.linear_equation(i, weights, c)
	cf_slope[index] = F.MSE(heights, y)

plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 12})
color = ['red','orange','black','green','blue','purple']
plt.scatter(m, cf_slope, s=300, c=color, edgecolors='black', alpha=1.0, marker='o')
plt.title('Cost Function with Constant Intercept', fontweight='bold')
plt.xlabel('slope', fontweight='bold')
plt.ylabel('MSEs', fontweight='bold')
plt.savefig('../drawing/image_out/' + 'cf' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()
