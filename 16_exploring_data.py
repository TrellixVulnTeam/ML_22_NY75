import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------------------#
# pip install seaborn
# pip install xgboost
#------------------------------------------------------------------------------------------#

# NOTE import file
filepath = '../datasets/save_tabular/data_20200107.csv'
df = pd.read_csv(filepath)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# NOTE create plot
sns.set_context('paper')
g1 = sns.countplot(x=df['crop_types'], palette='magma')
g1.set(xlabel=None)  # remove the axis label
labels = ['cassava', 'rice', 'maize', 'sugarcrane']
x = [0, 1, 2, 3]
plt.xticks(x, labels)
plt.ylabel('counts')
plt.title('Crop Types')
plt.show()