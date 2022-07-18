import pandas as pd
import matplotlib.pyplot as plt

'''
step 1: import data
'''

# data = pd.read_csv('../datasets/welllog_csv/welllogs.csv')
# print(data.describe())
# x = data['DeltaPHI']
# x = x.to_numpy()

# NOTE plot
# plt.figure(figsize=(20, 20))
# plt.scatter(data['DeltaPHI'], data['PE'], color='orange', linewidth=1, edgecolor='black', s=30)
# facies = data['Facies']
# plt.scatter(data['GR'], data['ILD_log10'], color='green', s=20, linewidth=1, edgecolor='black')
# plt.show()

'''
step 2: data transformation
'''
import numpy as np

ILD = np.array([0.664, 0.661, 0.658, 0.655, 0.647])
PHI = np.array([9.9, 14.2, 14.8, 13.9, 13.5])

data = np.concatenate((ILD, PHI))
# print(len(data))
# print(data.shape)
print(data)
