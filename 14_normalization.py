import sys
sys.path.append('./Libs') 
import basic_functions as bf
#-----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
# pip install sklearn
# pip install seaborn
#-----------------------------------------------------------------------------------------#

'''
step 1: plot input data
'''

data = pd.read_csv('save_tabular/demo_DT.csv')
# bf.scatter_plot_2(data['B2'])

'''
step 2: normalized data
'''

y = data['B2'].to_numpy().reshape(-1, 1)
scaler = RobustScaler()
scaler_1 = scaler.fit_transform(y)
scaler = MinMaxScaler(feature_range=(-2, 2))
scaler_2 = scaler.fit_transform(y)
nor = np.hstack([scaler_1, scaler_2])
bf.KDE_plot(nor)