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
# print(shape_file['crop_type']) # view specific column
# NOTE import shape file
header_information = rasterio.open('../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2').profile
# print(header_information)
src = rasterio.open('../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2')

# NOTE plot
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), tight_layout=True)
facies_colors = ['#27AE60', '#D4AC0D', '#D35400', '#9B59B6']
crop_colors = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
shape_file.plot(ax=ax, column='crop_type', cmap=crop_colors, legend=False)
show(src.read(), transform=src.transform, ax=ax, cmap='gray')
ax.set_title('Study Area', fontsize=14, fontweight='bold')
ax.set_xlabel('latitude', fontsize=12, fontweight='bold')
ax.set_ylabel('longtitude', fontsize=12, fontweight='bold')
ax.ticklabel_format(useOffset=False, style='plain')
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=12, weight='bold')
# plt.savefig('pictures/demo' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()