import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio

from rasterio.plot import show

shape_file = gpd.read_file('../datasets/shape_files/traindata.shp')
src = rasterio.open('../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B03.jp2')

# NOTE plot
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20), tight_layout=True)
facies_colors = ['#27AE60', '#D4AC0D', '#D35400', '#9B59B6']
crop_colors = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
shape_file.plot(ax=ax, column='crop_type', cmap=crop_colors, legend=False)
show(src.read(), transform=src.transform, ax=ax, cmap='gray')
ax.set_title('Study Area', fontsize=24, fontweight='bold')
ax.set_xlabel('latitude', fontsize=22, fontweight='bold')
ax.set_ylabel('longtitude', fontsize=22, fontweight='bold')
ax.ticklabel_format(useOffset=False, style='plain')
plt.xticks(fontsize=22, weight='bold')
plt.yticks(fontsize=22, weight='bold')
plt.savefig('pictures/demo' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()