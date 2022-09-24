import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

# NOTE spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=42, stratify=y)
print(X_train.head())
print(X_test.head())