import sys
sys.path.append('./Libs') 
import customized_functions as cf
#-----------------------------------------------------------------------------------------#
import pandas as pd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

'''
scatter plot
'''


# NOTE import B2 (blue)
B2_file = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2'
B2 = rasterio.open(B2_file)
B2 = B2.read(); B2 = B2[0, :, :]
B2[np.isnan(B2)] = 0 # replace nan with zeros (numpy.ndarray)
B2_vector = B2.flatten()
# B2 = B2[B2 != 0]
# print(B2)

# NOTE read label from csv
# df_1 = pd.read_csv('save_tabular/shape_file.csv')
# NOTE allocate dictionary
data = {
		'B2_vector'           : [],
        'B2_cassava'   : [],
		'B2_rice'      : [],
		'B2_maize'     : [],
		'B2_sugarcrane': []
        }
df = pd.DataFrame(data)
df['B2_vector'] = B2_vector
# print(df)

img_profile = rasterio.open(B2_file).profile
B2_shape = gpd.read_file('../datasets/shape_files/traindata.shp').to_crs(img_profile['crs'])
for i in range (0, len(B2_shape)):
	# print(i)
	farm = rasterize(shapes=(B2_shape.geometry[i], int(B2_shape.crop_type[i])), 
							out_shape=(img_profile['width'], img_profile['height']),
							transform=img_profile['transform'])
	farm = farm.flatten()
	if int(B2_shape.crop_type[i]) == 1:
		df['B2_cassava'] = farm
		# print('yes')
	elif int(B2_shape.crop_type[i]) == 2:
		df['B2_rice'] = farm
	elif int(B2_shape.crop_type[i]) == 3:
		df['B2_maize'] = farm
	elif int(B2_shape.crop_type[i]) == 4:
		df['B2_sugarcrane'] = farm
# print(df)
# df.to_csv('test.csv', index=False)
# farm = farm[farm_id != 0]
a = df['B2_vector']
# a = a[::10000]
b = df['B2_cassava']
# b = b[::10000]
y = np.linspace(0, len(a), len(a))
plt.scatter(y, a, color='orange')
plt.scatter(y, b, color='blue')
plt.ylim(250, a.max())
plt.show()

# plt.scatter(B2, color='orange')
# # NOTE plot
# _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20), tight_layout=True)
# facies_colors = ['#27AE60', '#D4AC0D', '#D35400', '#9B59B6']
# crop_colors = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
# shape_file.plot(ax=ax, column='crop_type', cmap=crop_colors, legend=False)
# show(src.read(), transform=src.transform, ax=ax, cmap='gray')
# ax.set_title('Study Area', fontsize=24, fontweight='bold')
# ax.set_xlabel('latitude', fontsize=22, fontweight='bold')
# ax.set_ylabel('longtitude', fontsize=22, fontweight='bold')
# ax.ticklabel_format(useOffset=False, style='plain')
# plt.xticks(fontsize=22, weight='bold')
# plt.yticks(fontsize=22, weight='bold')
# # plt.savefig('pictures/demo' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()