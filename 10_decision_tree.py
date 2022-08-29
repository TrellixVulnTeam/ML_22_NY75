import sys
sys.path.append('./Libs') 
import basic_functions as bf
#-----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#-----------------------------------------------------------------------------------------#

# TODO import data
data = pd.read_csv('../datasets/welllog_csv/welllogs.csv')

# NOTE prepare data
data = data[::10] # select every 10 data points 
# customize plot colors
lithocolors = ['#F4D03F',
               '#F5B041',
               '#DC7633',
               '#6E2C00',
               '#1B4F72',
               '#2E86C1',
               '#AED6F1',
               '#A569BD',
               '#196F3D']
# select well log to visualize
x = data['GR']
y = data['ILD_log10']
labels = data['Facies']
# assign label names to match with the lithocolors
lithofacies = ['SS',
               'CSiS',
               'FSiS',
               'SiSh',
               'MS',
               'WS',
               'D',
               'PS',
               'BS']
# bf.scatter_plot(lithocolors, x, y, labels, lithofacies)

# NOTE compute decision trees
title = 'Decision Tree (Max Depth: 20)'
ytitle = data.columns[5]
xtitle = data.columns[4]
bf.visualize_classifier(DecisionTreeClassifier(max_depth=20), lithocolors,
                        data['GR'], data['ILD_log10'], lithofacies,
                        data['Facies'],
                        title, ytitle, xtitle,
                        )