import sys
sys.path.append('./Libs') 
import customized_functions as cf
#-----------------------------------------------------------------------------------------#
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import numpy as np

'''
Part 2: part 1 demonstates how to collect satellite image where farms are located into dictionary. This section will use for-loops to collect data into csv. 
'''

'''
step 1:
'''
# NOTE read label from csv
df_1 = pd.read_csv('save_tabular/shape_file.csv')
# NOTE allocate dictionary
data = {
        'crop_types': [],
        'B2_mean'   : [],
		'B2_median' : []
        }
df = pd.DataFrame(data)
df['crop_types'] = df_1['crop_type']

'''
step 2: extract farm ids (small images)
'''

# NOTE import satellite image and shape file
file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2'
img_profile = rasterio.open(file_name).profile
shape_file = gpd.read_file('../datasets/shape_files/traindata.shp').to_crs(img_profile['crs'])
image = rasterio.open(file_name)
image = image.read(); image = image[0, :, :]
image[np.isnan(image)] = 0 # replace nan with zeros (numpy.ndarray)
# NOTE demo --> extract 1 farm id (10)

for i in range (0, len(shape_file)):
	# print(shape_file.crop_type[i])
	farm = rasterize(shapes=(shape_file.geometry[i], int(shape_file.crop_type[i])), 
							out_shape=(img_profile['width'], img_profile['height']),
							transform=img_profile['transform'])
	farm_id = cf.extract_farm_id(image, farm)
	plt.imshow(farm_id)
	plt.show()