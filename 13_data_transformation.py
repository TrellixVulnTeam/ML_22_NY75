import sys
sys.path.append('./Libs') 
import functions as f
#-----------------------------------------------------------------------------------------#
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.features import rasterize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

'''
step 1: import metadata
'''

# NOTE path file
file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2'
# NOTE read in vector
vector = gpd.read_file('../datasets/shape_files/traindata.shp')
# NOTE get list of geometries for all features in vector file
# geom = [shapes for shapes in vector.geometry]
# NOTE convert polygon to raster
raster = rasterio.open(file_name)
# NOTE rasterize vector using the shape and coordinate system of the raster
rasterized = features.rasterize(shapes=(vector.geometry[220], int(vector.crop_type[220])),
                                out_shape = raster.shape,
                                transform = raster.transform)
# NOTE plot
# plt.figure(figsize=(10, 10))
# plt.imshow(rasterized)
# plt.gca().invert_yaxis()
# plt.savefig('../drawing/image_out/' + 'rasterized' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()

'''
step 2: the previous step provided one farm without intensity and step 2 will patch the intensity into the extracted farm id.
'''

# NOTE read in satellite image and slice out the header
raster = raster.read(1)
# NOTE check nan and replace with zeros
raster[np.isnan(raster)] = 0 
# NOTE crop only interested farm
farm_id = f.extract_farm_id(raster, rasterized, 'no')
# NOTE plot
# plt.figure(figsize=(10, 10))
# plt.imshow(farm_id[1520:1550, 710:740], interpolation=None)
# plt.gca().invert_yaxis()
# plt.savefig('../drawing/image_out/' + 'patch' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()

'''
step 3: transform vector to 2D matrix
'''

B2 = farm_id[farm_id != 0.].reshape(-1, 1)
crop_type = np.repeat(vector.crop_type[220], len(B2)).reshape(-1, 1)
B2_median = np.repeat(np.median(B2), len(B2)).reshape(-1, 1)
B2_mean = np.repeat(np.mean(B2), len(B2)).reshape(-1, 1)
data_np = np.hstack((crop_type, B2, B2_median, B2_mean))
# print(data_np)

'''
step 4: optional, in case we want to keep data as a dataframe or csv
'''

# dict = {
#         'crop_types': [],
#         'B2': [],
#         'B2_median': [],
#         'B2_mean': []
#         }
# df = pd.DataFrame(dict)
# df['crop_types'] = data_np[:, 0]
# df['B2'] = data_np[:, 1]
# df['B2_median'] = data_np[:, 2]
# df['B2_mean'] = data_np[:, 3]
# # print(df)
# # df.to_csv ('save_tabular/demo_DT.csv', index = None, header=True) 