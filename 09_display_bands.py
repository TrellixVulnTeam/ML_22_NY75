import sys
sys.path.append('./Libs') 
import customized_functions as C
#-----------------------------------------------------------------------------------------#
import rasterio

# NOTE import near infrared band (B8)
file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B08.jp2'
nir = rasterio.open(file_name)
nir = nir.read(); nir = nir[0, :, :]
# min_num, max_num = C.clip(nir, 100)
# C.plot_band(nir, min_num, max_num)

# NOTE import red band (B4)
file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B04.jp2'
red = rasterio.open(file_name)
red = red.read(); red = red[0, :, :]
# min_num, max_num = C.clip(red, 99)
# C.plot_band(red, min_num, max_num)

# NOTE NDVI
ndvi = C.NDVI(nir, red)
# min_num, max_num = C.clip(ndvi, 95)
# C.plot_band(ndvi, min_num, max_num)
# C.histogram_plot(ndvi)

# NOTE import red band (B3)
file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2'
B3 = rasterio.open(file_name)
B3 = B3.read(); B3 = B3[0, :, :]
file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2'
B2 = rasterio.open(file_name)
B2 = B2.read(); B2 = B2[0, :, :]
C.plot_rgb(B2, B3, red, 1)
