import pandas as pd

# NOTE read label from csv
df_1 = pd.read_csv('save_tabular/shape_file.csv')
# print(df_1)
# NOTE allocate dictionary
data = {
        'crop_types': [],
        'B2_mean'   : [],
		'B2_median' : []
        }

# print(data)
df = pd.DataFrame(data)
# print(type(df))
# df.to_csv('save_tabular/test.csv', index=False)
df['crop_types'] = df_1['crop_type']
# print(df)
test = [45, 11, 2, 113, 29]
for i in range (0, 5):
	df['B2_mean'][0] = test[i]
print(df)