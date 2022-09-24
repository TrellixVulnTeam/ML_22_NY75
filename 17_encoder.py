import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------------------#

# NOTE import file
filepath = '../datasets/save_tabular/data_20200107.csv'
df = pd.read_csv(filepath)

# NOTE encoder
le = LabelEncoder()
y = le.fit_transform(df['crop_types']) # output numpy array
X = df.drop('crop_types', axis=1) # output dataframe
print(X.head())
# print(y[0:10000:100])
# print(np.unique(y))

# NOTE sort crop types
rice = df.loc[df['crop_types'] == 2]
print(rice)

# NOTE old way
# y[y == 1] = 0
# y[y == 2] = 1
# y[y == 3] = 2
# y[y == 4] = 3