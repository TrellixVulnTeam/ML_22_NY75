import sys
sys.path.append('./Libs') 
import customized_functions as cf
#-----------------------------------------------------------------------------------------#
import rasterio
from skimage.transform import resize
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
import matplotlib.pyplot as plt
# pip install scikit-image
#-----------------------------------------------------------------------------------------#

# NOTE import red band (B4)
# file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2'
# red = rasterio.open(file_name)
# red = red.read(); red = red[0, :, :]
# min_num, max_num = cf.clip(red, 99)
# cf.plot_band(red, 'Reds', min_num, max_num, 'B4')
# red = red[red != 0]
# cf.histogram_plot(red)

# NOTE import satellite image and shape file
file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B04.jp2'
img_profile = rasterio.open(file_name).profile
shape_file = gpd.read_file('../datasets/shape_files/traindata.shp').to_crs(img_profile['crs'])
image = rasterio.open(file_name)
image = image.read(); image = image[0, :, :]
image[np.isnan(image)] = 0 # replace nan with zeros (numpy.ndarray)
# NOTE demo --> extract 1 farm id (10)
farm = rasterize(shapes=(shape_file.geometry[220], int(shape_file.crop_type[220])), 
						out_shape=(img_profile['width'], img_profile['height']),
						transform=img_profile['transform'])
farm_id = cf.extract_farm_id(image, farm, 'yes')
print(int(shape_file.crop_type[220]))
farm_id = farm_id[farm_id != 0]
cf.histogram_plot(farm_id)
# plt.imshow(farm_id)
# plt.show()