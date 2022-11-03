import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------------------#

'''
step 1: loops paramters
'''

# NOTE import file
filepath = '../datasets/save_tabular/data_all_2020.csv'
df = pd.read_csv(filepath)

# NOTE encoder
le = LabelEncoder()
y = le.fit_transform(df['crop_types'])
X = df.drop('crop_types', axis=1)

# NOTE spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# NOTE fit matrix form
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test)

# NOTE define parameters
cost_function = np.zeros(shape=(4, 4), dtype=np.float)
max_depth = [9, 15, 20, 35]
learning_rate = [1, 0.1, 0.15, 0.2]
params = {
          'objective': 'multi:softmax',
          'max_depth': [],
          'learning_rate': [],
          'num_class': 4
         }
for i in range (0, len(max_depth)):
	params['max_depth'] = max_depth[i]
	for ii in range (0, len(learning_rate)):
		params['learning_rate'] = learning_rate[ii]
		# print(params)
		bst = xgb.train(params, dtrain)
		pred = bst.predict(dtest)
		f1_ = f1_score(y_test, pred, average='weighted')
		print('f1: %.5f' % f1_, 'max depth: ', max_depth[i], 'learning rate: ', learning_rate[ii])
		cost_function[i, ii] = f1_
print(cost_function)

'''
step 2: plot cost function using smooth contours
'''

fig = go.Figure(data = go.Contour(z=cost_function, colorscale='Electric'))
# fig.write_image('pictures/smooth_color.svg', format='svg')
fig.show()
