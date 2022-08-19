import sys
sys.path.append('./Libs') 
import functions as cf
#-----------------------------------------------------------------------------------------#
import pandas as pd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

'''
Part 2: This part initiates for-loop to transform farm ids into a data point in tabular data. We demonstrate only band 2. 
'''

'''
step 1: create dictionary and convert to dataframe
'''

# NOTE create dictionary
B2_list = ['B2_vector', 'B2_cassava', 'B2_rice', 'B2_maize', 'B2_sugarcrane']
data = {}
allocate_list = []
for i in B2_list:
    data[i] = allocate_list
# data = {
# 		'B2_vector'    : [],
#       'B2_cassava'   : [],
# 		'B2_rice'      : [],
# 		'B2_maize'     : [],
# 		'B2_sugarcrane': []
#         }
df = pd.DataFrame(data) # convert to dataframe
# NOTE import B2 (blue)
B2_file = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2'
B2 = rasterio.open(B2_file)
B2 = B2.read(); B2 = B2[0, :, :]
B2[np.isnan(B2)] = 0 # replace nan with zeros (numpy.ndarray)
B2_vector = B2.flatten()
df['B2_vector'] = B2_vector # add B2 to dataframe

'''
step 2: import raster and shape file 
'''

img_profile = rasterio.open(B2_file).profile
B2_shape = gpd.read_file('../datasets/shape_files/traindata.shp').to_crs(img_profile['crs'])
# NOTE pre-allocate memory
cassava = np.zeros(shape=len(df), dtype=np.float32)
rice = np.zeros(shape=len(df), dtype=np.float32)
maize = np.zeros(shape=len(df), dtype=np.float32)
sugarcrane = np.zeros(shape=len(df), dtype=np.float32)

'''
step 3: create for-loop to correct farm ids where belong to crops (1354 farms)
'''

for i in range (0, len(B2_shape)):
	farm = rasterize(shapes=(B2_shape.geometry[i], int(B2_shape.crop_type[i])), 
							 out_shape=(img_profile['width'], img_profile['height']),
							 transform=img_profile['transform'])
	farm = cf.extract_farm_id(B2, farm, 'no')
	farm = farm.flatten()
	if int(B2_shape.crop_type[i]) == 1:
		cassava += farm 
	elif int(B2_shape.crop_type[i]) == 2:
		rice += farm 
	elif int(B2_shape.crop_type[i]) == 3:
		maize += farm
	elif int(B2_shape.crop_type[i]) == 4:
		sugarcrane += farm
df['B2_cassava'] = cassava
df['B2_rice'] = rice
df['B2_maize'] = maize
df['B2_sugarcrane'] = sugarcrane
df.to_csv('save_tabular/B2_scatter.csv', index=False)

'''
step 4: scatter plot
'''

a = df['B2_vector']
a = a[::1000]
b = df['B2_cassava']
b = b[::1000]
c = df['B2_rice']
c = c[::1000]
d = df['B2_maize']
d = d[::1000]
e = df['B2_sugarcrane']
e = e[::1000]
y = np.linspace(0, len(a), len(a))
# NOTE scatter plot
plt.figure(figsize=(20, 10))
plt.scatter(y, a, color='blue', s=20, edgecolor='black')
plt.scatter(y, b, color='#27AE60', s=100, edgecolor='black')
plt.scatter(y, c, color='#D4AC0D', s=100, edgecolor='black')
plt.scatter(y, d, color='#D35400', s=100, edgecolor='black')
plt.scatter(y, e, color='#9B59B6', s=100, edgecolor='black')

# plt.scatter(y, b, color='blue', s=100, edgecolor='black')
# plt.scatter(y, c, color='blue', s=100, edgecolor='black')
# plt.scatter(y, d, color='blue', s=100, edgecolor='black')
# plt.scatter(y, e, color='blue', s=100, edgecolor='black')
plt.ylim(250, 1380)
plt.axis('off')
plt.savefig('pictures/scatter_labels' + '.png', format='png', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()
