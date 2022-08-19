import sys
sys.path.append('./Libs') 
import functions as f
#-----------------------------------------------------------------------------------------#
import rasterio
from skimage.transform import resize
# pip install scikit-image
#-----------------------------------------------------------------------------------------#

'''
step 1: learn a simple plot
'''

# NOTE import red band (B4)
# file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2'
# red = rasterio.open(file_name)
# red = red.read(); red = red[0, :, :]
# min_num, max_num = cf.clip(red, 99)
# cf.plot_band(red, 'Blues', min_num, max_num, 'B4')

'''
step 2: for-loop plots for single band 
'''

# NOTE file names
# B4 = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B04.jp2'
# B3 = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2'
# B2 = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2'
# bands = [B4, B3, B2]
# # NOTE band color
# band_color = ['Reds', 'Greens', 'Blues']
# # NOTE save names
# save_band = ['B4', 'B3', 'B2']
# # NOTE begin for-loop
# for index, i in enumerate(bands):
# 	print('accessing band: ', i)
# 	band = rasterio.open(i)
# 	band = band.read(); band = band[0, :, :]
# 	min_num, max_num = f.clip(band, 99)
# 	f.plot_band(band, band_color[index], min_num, max_num, save_band[index])

'''
step 3: for-loop plots for composite colors
'''

# NOTE pre-load path files
file_part = '../datasets/sentinel_2/2020/20200107/IMG_DATA/'
B1  = file_part + '47PQS_20200107_B01.jp2'
B2  = file_part + '47PQS_20200107_B02.jp2'
B3  = file_part + '47PQS_20200107_B03.jp2'
B4  = file_part + '47PQS_20200107_B04.jp2'
B8  = file_part + '47PQS_20200107_B08.jp2'
B8A = file_part + '/47PQS_20200107_B8A.jp2'
B11 = file_part + '/47PQS_20200107_B11.jp2'
B12 = file_part + '/47PQS_20200107_B12.jp2'
bands = [B1, B2, B3, B4, B8, B8A, B11, B12]
# NOTE for-loop for storing each band in variables
for index, i in enumerate(bands):
	print('accessing band: ', i)
	band = rasterio.open(i)
	band = band.read(); band = band[0, :, :]
	if index == 0:
		B1 = band
	elif index == 1:
		B2 = band
	elif index == 2:
		B3 = band
	elif index == 3:
		B4 = band
	elif index == 4:
		B8 = band
	elif index == 5:
		B8A = band
	elif index == 6:
		B11 = band
	elif index == 7:
		B12 = band
# NOTE natural colors (B4, B3, B2)
# f.composite_bands(B4, B3, B2, 0.95)
# NOTE color infrared (B8, B4, B3)
# f.composite_bands(B8, B4, B3, 0.90)
# NOTE short-wave infrared (B12, B8A, B4)
resized_B4 = resize(B4, (B12.shape[0], B12.shape[1])) # type: ignore 
f.composite_bands(B12, B8A, resized_B4, 0.90)
# NOTE agriculture (B11, B8, B2)
resized_B2 = resize(B2, (B11.shape[0], B11.shape[1])) # type: ignore
resized_B8 = resize(B8, (B11.shape[0], B11.shape[1])) # type: ignore 
f.composite_bands(B11, resized_B8, resized_B2, 0.90)
# NOTE geology (B12, B11, B2)
f.composite_bands(B12, B11, resized_B2, 0.95)
f.composite_bands(resized_B2, B11, B12, 0.95)
# NOTE bathymetric (B4, B3, B1)
resized_B4 = resize(B4, (B1.shape[0], B1.shape[1])) # type: ignore 
resized_B3 = resize(B3, (B1.shape[0], B1.shape[1])) # type: ignore 
f.composite_bands(resized_B4, resized_B3, B1, 0.90)
# NOTE vegetation index (B8-B4)/(B8+B4)
veg = f.NDVI(B8, B4)
min_num, max_num = f.clip(veg, 95)