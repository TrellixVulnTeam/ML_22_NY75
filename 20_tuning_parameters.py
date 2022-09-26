import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pretty_confusion_matrix import pp_matrix_from_data
#------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
# pip install pretty-confusion-matrix
#------------------------------------------------------------------------------------------#

# NOTE import file
filepath = '../datasets/save_tabular/data_20200107.csv'
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
params = {
          'objective': 'multi:softmax',
          'max_depth': 3,
          'alpha': 10,
          'learning_rate': 0.01,
          'num_class': 4
         }

# NOTE training
bst = xgb.train(params, dtrain)
pred = bst.predict(dtest)
# print(pred)

# NOTE plot confusion matrix
pp_matrix_from_data(y_test, pred, cmap='Dark2')