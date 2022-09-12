import sys
sys.path.append('./Libs') 
import basic_functions as bf
#-----------------------------------------------------------------------------------------#
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.colors as colors
#-----------------------------------------------------------------------------------------#
​
'''
step 1: select your farm
'''
​
# NOTE import satellite image and shape file
file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2'
img_profile = rasterio.open(file_name).profile
shape_file = gpd.read_file('../datasets/shape_files/traindata.shp').to_crs(img_profile['crs'])
image = rasterio.open(file_name)
image = image.read(); image = image[0, :, :]
​
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
rect = patches.Rectangle((1180, 1010), (1225-1180), (1035-1010), linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()
​
'''
step 2: crop only your farm
'''
​
crop_image = image[1010:1035, 1180:1225]
plt.imshow(crop_image, cmap='gray')
plt.show()
​
'''
step 3: compute kmeans
'''
​
x = crop_image.flatten()
y = np.linspace(0, len(x), len(x))
number_of_classes = 2
a = bf.compute_kmeans(x, y, number_of_classes)
a = np.unique(a)
​
'''
step 4: make new image for kmeans
'''
​
new_map = np.zeros_like(crop_image, dtype=float)
for i in range (0, new_map.shape[0]):
	for ii in range (0, new_map.shape[1]):
		if crop_image[i, ii] >= a[0]:
			new_map[i, ii] = crop_image[i, ii]
cmap = colors.ListedColormap(['#05368C', '#96FF33'])
plt.imshow(new_map, cmap=cmap)
plt.show()