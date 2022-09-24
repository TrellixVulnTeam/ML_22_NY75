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
from numpy import save
#-----------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
#-----------------------------------------------------------------------------------------#

'''
We use time stamp 07/01/2020 and four bands namely, B2, B3, B4, and B8.
Step 1: data transformation, rasterize farm ids into vectors for each band.
'''

# NOTE path file
file_20200107 = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_'
bands = ['B02.jp2', 'B03.jp2', 'B04.jp2', 'B08.jp2']

vector = gpd.read_file('../datasets/shape_files/traindata.shp')
number_of_farm = len(vector)
for i in bands:
	raster = rasterio.open(file_20200107+i) # read metadata 
	v_data = np.array([0, 0, 0, 0])
	for ii in range (0, number_of_farm):
		rasterized = features.rasterize(shapes=(vector.geometry[ii], int(vector.crop_type[ii])),
										out_shape = raster.shape,
										transform = raster.transform)
		farm_id = f.extract_farm_id(raster.read(1), rasterized, 'no')
		h_data = f.hstack_farm(farm_id, int(vector.crop_type[ii]))
		v_data = np.vstack((v_data, h_data))
	v_data = v_data[1:, :]
	save_file_name = 'save_npy/' + i.removesuffix('.jp2') + '_20200107' + '.npy'
	print('saving: ', save_file_name)
	save(save_file_name, v_data)

'''
step 2: load binary numpy array and convert to csv (ASCII) format
'''

data_20200107 = np.hstack((np.load('save_npy/B02_20200107.npy'),
						   np.load('save_npy/B03_20200107.npy'),
						   np.load('save_npy/B04_20200107.npy'),
						   np.load('save_npy/B08_20200107.npy')))

dict = {
        'crop_type_B2': [],
        'B02_20200107': [],
        'B02_20200107_median': [],
        'B02_20200107_mean': [],
        'crop_type_B3': [],
        'B03_20200107': [],
        'B03_20200107_median': [],
        'B03_20200107_mean': [],
        'crop_type_B4': [],
        'B04_20200107': [],
        'B04_20200107_median': [],
        'B04_20200107_mean': [],
        'crop_type_B8': [],
        'B08_20200107': [],
        'B08_20200107_median': [],
        'B08_20200107_mean': []
        }

df = pd.DataFrame(dict)
for id_, i in enumerate(df):
	df[i] = data_20200107[:, id_]
df.to_csv ('save_tabular/data_20200107.csv', index=None, header=True) 