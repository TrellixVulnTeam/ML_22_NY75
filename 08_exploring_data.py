import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio

from rasterio.plot import show
# pip install rasterio
# pip install geopandas

shape_file = gpd.read_file('../datasets/shape_files/traindata.shp')
src = rasterio.open('../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2')
# print(shape_file.head())
# shape_file = shape_file[['crop_type']]
# print(shape_file)

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
facies_colors = ['#27AE60', '#D4AC0D', '#D35400', '#9B59B6']
crop_colors = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
shape_file.plot(ax=ax, column='crop_type', cmap=crop_colors, legend=True)
show(src.read(), transform=src.transform, ax=ax, cmap='gray')
plt.show()