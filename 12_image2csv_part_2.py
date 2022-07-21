import sys
sys.path.append('./Libs') 
import customized_functions as cf
#-----------------------------------------------------------------------------------------#
import pandas as pd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

'''
Creating scatter plot that overleys with labels (crop types)
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
a = a[::100]
b = df['B2_cassava']
b = b[::100]
c = df['B2_rice']
c = c[::100]
d = df['B2_maize']
d = d[::100]
e = df['B2_sugarcrane']
e = e[::100]
y = np.linspace(0, len(a), len(a))
plt.scatter(y, a, color='orange')
plt.scatter(y, b, color='red')
plt.scatter(y, c, color='black')
plt.scatter(y, d, color='blue')
plt.scatter(y, e, color='green')
plt.ylim(250, a.max())
plt.show()
