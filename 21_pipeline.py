import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pretty_confusion_matrix import pp_matrix_from_data
import matplotlib.pylab as plt
#------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
# pip install pretty-confusion-matrix
#------------------------------------------------------------------------------------------#

# NOTE import file
# filepath = '../datasets/save_tabular/data_20200107.csv'
filepath = '../datasets/save_tabular/data_all_2020.csv'
df = pd.read_csv(filepath)

# NOTE encoder
le = LabelEncoder()
y = le.fit_transform(df['crop_types'])
X = df.drop('crop_types', axis=1)

# NOTE spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# NOTE define parameters for fitting
clf_xgb = xgb.XGBClassifier(base_score=0.5,
                            booster='gbtree',
                            learning_rate=0.01,
                            n_estimators=500,
                            objective='multi:softprob',
                            # subsample=1, tree_method='gpu_hist', gpu_id=0,
                            subsample=1, 
                            verbosity=1)
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=50,
            eval_metric='merror',
            eval_set=[(X_train, y_train), (X_test,y_test)])

# NOTE plot learning curves
results = clf_xgb.evals_result()
plt.plot(results['validation_0']['merror'], label='train')
plt.plot(results['validation_1']['merror'], label='test')
plt.legend()
plt.show()

# NOTE print f1 score 
y_pred = clf_xgb.predict(X_test)
f1_ = f1_score(y_test, y_pred, average='weighted')
print('f1 score weighted: %.3f' % f1_)

# NOTE plot confusion matrix
pp_matrix_from_data(y_test, y_pred, cmap='tab10')

# NOTE feature importance
print(clf_xgb.feature_importances_)
plt.figure(figsize=(6, 12))
plt.bar(range(len(clf_xgb.feature_importances_)), clf_xgb.feature_importances_)
labels = df.columns[1:]
# print(labels)
x = np.arange(0, len(labels), 1)
plt.xticks(x, labels, rotation=90)
plt.ylabel('values (the more is the better)')
plt.title('Feature Importances', fontweight='bold')
plt.show()