import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio

from rasterio.plot import show
# pip install rasterio
# pip install geopandas
#-----------------------------------------------------------------------------------------#

'''
Satellite imagery is metadata containing arrays, georeference, indexes, and polygons, which uses geopandas to manipulate shape files. In this exercise, we will explore farm boundaries containing four crop types that embed in georeference. To fit the image (array) in latitude and longitude, we use rasterio and geopandas to manipulate the header and array. 
'''

# NOTE import shape file
shape_file = gpd.read_file('../datasets/shape_files/traindata.shp') # contain tabular data
# print(shape_file.head()) # print the first five rows
# shape_file = shape_file[['crop_type']]
# print(shape_file) # view specific column
# NOTE import shape file
# header_information = rasterio.open('../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2').profile
# print(header_information)

# NOTE plot
arrays = rasterio.open('../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2')
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
facies_colors = ['#27AE60', '#D4AC0D', '#D35400', '#9B59B6']
crop_colors = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
shape_file.plot(ax=ax, column='crop_type', cmap=crop_colors, legend=True)
show(arrays.read(), transform=arrays.transform, ax=ax, cmap='viridis')
plt.show()