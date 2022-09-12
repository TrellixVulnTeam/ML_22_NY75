import sys
sys.path.append('./Libs') 
import basic_functions as bf
#-----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#-----------------------------------------------------------------------------------------#

'''
step 1: compute outliers
'''

data = pd.read_csv('save_tabular/demo_DT.csv')
bf.scatter_plot_2(data['B2'])

y = data['B2'].to_numpy().reshape(-1, 1)
print(np.quantile(y, 0.01))
print(np.quantile(y, 0.5))
print(np.quantile(y, 0.99))

'''
step 2: mark outlier points
'''

fig, ax = plt.subplots()
# NOTE seaborn row-major axis
sns.histplot(data=data['B2'], color = 'orange', alpha = 0.6, kde = True, line_kws = {'color':'red','linestyle': 'dashed'}, label='Quantile', ax=ax)
ax.plot([np.quantile(y, 0.01), np.quantile(y, 0.01)], [0, 65], '-r', linewidth=2)
ax.plot([np.quantile(y, 0.5), np.quantile(y, 0.5)], [0, 65], '-r', linewidth=2)
ax.plot([np.quantile(y, 0.99), np.quantile(y, 0.99)], [0, 65], '-r', linewidth=2)
plt.title('Kernel Density Estimation (KDE)', fontweight='bold')
plt.xlabel('intensity'); plt.ylabel('count')
plt.legend(loc='upper left')
# plt.savefig('../drawing/image_out/' + 'out' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()