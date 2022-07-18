import sys
sys.path.append('./Libs') 
import customized_functions as cf
#-----------------------------------------------------------------------------------------#
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
# import matplotlib.colors as colors

# from rasterio.plot import show

'''
step 1: create csv 
'''

df_1 = pd.read_csv('save_tabular/shape_file.csv')
# print(df_1)

# initialize data of lists.
data = {
        'crop_types': [],
        'B2'        : [],
		'B3'        : [],
        'B4'        : [],
        'B8'        : []
        }
# Create DataFrame
df = pd.DataFrame(data)
# Print the output.
# print(df)
df['crop_types'] = df_1['crop_type']
# print(df)

'''
step 2: correcting data from farm ids
'''

file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2'
img_profile = rasterio.open(file_name).profile
shape_file = gpd.read_file('../datasets/shape_files/traindata.shp').to_crs(img_profile['crs'])

image = rasterio.open(file_name)
image = image.read(); image = image[0, :, :]
# for i in range (0, len(df)):
# 	# print(df['crop_types'][i])
# 	# farm = rasterize(shapes=(shape_file.geometry[i], int(shape_file.crop_type[i])), 
# 	farm = rasterize(shapes=(shape_file.geometry[10], int(shape_file.crop_type[10])), 
# 								out_shape=(img_profile['width'], img_profile['height']),
# 								transform=img_profile['transform'])
# 	# print(type(farm))
# 	# plt.imshow(farm)
# 	# plt.show()
# 	farm = farm[farm != 0]
# 	print(farm)

farm = rasterize(shapes=(shape_file.geometry[10], int(shape_file.crop_type[10])), 
						 out_shape=(img_profile['width'], img_profile['height']),
						 transform=img_profile['transform'])
farm_id = cf.extract_farm_id(image, farm)
plt.imshow(farm_id)
plt.show()