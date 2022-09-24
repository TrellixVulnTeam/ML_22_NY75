import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import matplotlib.pylab as plt
#------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------------------#

# NOTE import file
filepath = '../datasets/save_tabular/data_20200107.csv'
df = pd.read_csv(filepath)

# NOTE encoder
le = LabelEncoder()
y = le.fit_transform(df['crop_types'])
X = df.drop('crop_types', axis=1)

# NOTE spliting data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# NOTE training data
evalset = [(X_train, y_train), (X_test,y_test)]
model = XGBClassifier()
model.fit(X_train, y_train, eval_metric='merror', eval_set=evalset)
# evaluate performance
y_pred = model.predict(X_test)
f1_ = f1_score(y_test, y_pred, average='weighted')
print('f1 score weighted: %.3f' % f1_)
# retrieve performance metrics
results = model.evals_result()

# NOTE plot learning curves
plt.plot(results['validation_0']['merror'], label='train')
plt.plot(results['validation_1']['merror'], label='test')
plt.legend()
plt.show()