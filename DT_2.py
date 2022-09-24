import sys
sys.path.append('./Libs') 
import functions as f
#-----------------------------------------------------------------------------------------#
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from numpy import save
from rasterio import features
from rasterio.features import rasterize
#-----------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
#-----------------------------------------------------------------------------------------#

'''
B2, B3, B4, B8, veg 
'''

'''
step 1: convert jp2 to npy
'''

# pathfile = '../datasets/sentinel_2/no_cloud/'
# dir_list = sorted(os.listdir(pathfile))
# # print(dir_list)

# # # NOTE keep
# # count = 1
# # for i in dir_list:
# # 	if count == 1:
# # 		file_ = pathfile + i + '/IMG_DATA/47PQS_' + i + '_'
# # 		f.access_bands(file_, i)
# # 		# print(file_)
# # 	else:
# # 		print('xxx')
# # 	count += 1

# for i in dir_list:
# 	file_ = pathfile + i + '/IMG_DATA/47PQS_' + i + '_'
# 	f.access_bands(file_, i)

'''
step 2: normalized data
'''

# pathfile = '../datasets/save_npy_2/'
# dir_list = sorted(os.listdir(pathfile))

# for i in dir_list:
# 	pathfile_i = pathfile + i
# 	nor_data = f.normalize_clip(np.load(pathfile_i))
# 	# plt.imshow(nor_data)
# 	# plt.show()
# 	save_file = '../datasets/normalized_npy/' + i.removesuffix('.npy') + '_nor' + '.npy'
# 	print('saving: ', save_file)
# 	save(save_file, nor_data)

'''
step 3: extracted farm ids
'''

# vector = gpd.read_file('../datasets/shape_files/traindata.shp')
# number_of_farm = len(vector)
# raster = rasterio.open('../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2')

# pathfile = '../datasets/normalized_npy/'
# dir_list = sorted(os.listdir(pathfile))

# for i in dir_list:
# 	v_data = np.array([0, 0]) # crop types, band
# 	for ii in range (0, number_of_farm):
# 		rasterized = features.rasterize(shapes=(vector.geometry[ii], int(vector.crop_type[ii])),
# 										out_shape = raster.shape,
# 										transform = raster.transform)
# 		farm_id = f.extract_farm_id(np.load(pathfile + i), rasterized, 'no')
# 		# plt.imshow(farm_id)
# 		# plt.show()
# 		h_data = f.hstack_farm_2(farm_id, int(vector.crop_type[ii]))
# 		v_data = np.vstack((v_data, h_data))
# 	v_data = v_data[1:, :]
# 	save_file = '../datasets/DT_npy/' + i.removesuffix('.npy') + '_DT' + '.npy'
# 	print('saving: ', save_file)
# 	save(save_file, v_data)

'''
step 4: load binary numpy array and convert to csv (ASCII) format
'''

dict = {}
pathfile = '../datasets/DT_npy/'
dir_list = sorted(os.listdir(pathfile))

for i in dir_list:
	dict.update({i.removesuffix('.npy'): []})

df = pd.DataFrame(dict)

for index, i in enumerate(dir_list):
	a = np.load(pathfile + i)
	df[i.removesuffix('.npy')] = a[:, -1]
label = np.load(pathfile + dir_list[0])
label = label[:, 0]
df.insert(0, 'crop_types', label)
df.to_csv('save_tabular/data_all_2020.csv', index=None, header=True) 