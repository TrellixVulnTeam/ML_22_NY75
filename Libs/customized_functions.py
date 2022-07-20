import matplotlib.pyplot as plt
import numpy as np

from skimage import exposure 

def clip(image, perc):
	''' 
	clip is useful 
	'''
	image = image.flatten()
	image = np.sort(image)
	vector_length = len(image)
	min_num, max_num = 0, 0
	if perc == 100:
		min_num = image[0]
		max_num = image[-1]
	elif perc != 100:
		min_num = image[int(vector_length*(1-(perc/100)))] 
		max_num = image[int(vector_length*(perc/100))]
	return min_num, max_num

def plot_band(image, min_num, max_num):
	'''
	General plot
	'''
	plt.figure(figsize=(20, 20))
	plt.imshow(image, cmap='terrain', vmin=min_num, vmax=max_num)
	plt.colorbar(orientation='vertical')
	plt.show()

def NDVI(nir, red):
	'''
	Sentinel-2
	NDVI = (NIR - RED) / (NIR + RED)
	RED is B4, 664.5 nm
	NIR is B8, 835.1 nm
	'''
	nir = np.nan_to_num(nir, nan=-1., posinf=1., neginf=-1.)
	red = np.nan_to_num(red, nan=-1., posinf=1., neginf=-1.)
	np.seterr(divide='ignore', invalid='ignore') # don't display the error of number/0
	output = (nir - red)/(nir + red)
	return output

def histogram_plot(image):
	'''
	histogram
	'''
	plt.figure(figsize=(20, 20))
	# image = image[image != 0]
	image = image[image <= 1]
	x = image[~np.isnan(image)]
	plt.hist(x.flatten(), bins=30, color='g', histtype='bar', ec='black')
	plt.show()

def normalized_data(y, lowest_value, highest_value):
	y = (y - y.min()) / (y.max() - y.min())
	return y * (highest_value - lowest_value) + lowest_value

def plot_rgb(B2, B3, B4, clip):
	'''
	RGB plot
	'''
	B2 = normalized_data(B2, 0, 1)
	B3 = normalized_data(B3, 0, 1)
	B4 = normalized_data(B4, 0, 1)
	# stackedRGB = np.stack((B2, B3, B4), axis=2)
	stackedRGB = np.stack((B4, B3, B2), axis=2)
	print(stackedRGB.shape)
	pLow, pHigh = np.percentile(stackedRGB[~np.isnan(stackedRGB)], (clip, 100-clip))
	stackedRGB = exposure.rescale_intensity(stackedRGB, in_range=(pLow, pHigh))
	plt.imshow(stackedRGB, cmap='terrain')
	plt.show()

def trim_zeros(arr):
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]

def extract_farm_id(image, farm_id, trim_zero):
	index_list = np.where(farm_id != 0)
	dummy = np.zeros(shape=(image.shape[0], image.shape[1]))
	for i in range (0, len(index_list[0])):
		dummy[index_list[0][i], index_list[1][i]] = image[index_list[0][i], index_list[1][i]]
	# NOTE trim farm
	if trim_zero == 'yes':
		dummy = trim_zeros(dummy)
	elif trim_zero == 'no':
		pass
	return dummy

# def scatter_plot():
# 	_, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20), tight_layout=True)
# 	facies_colors = ['#27AE60', '#D4AC0D', '#D35400', '#9B59B6']
# 	crop_colors = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
# 	shape_file.plot(ax=ax, column='crop_type', cmap=crop_colors, legend=False)
# 	show(src.read(), transform=src.transform, ax=ax, cmap='gray')
# 	ax.set_title('Study Area', fontsize=24, fontweight='bold')
# 	ax.set_xlabel('latitude', fontsize=22, fontweight='bold')
# 	ax.set_ylabel('longtitude', fontsize=22, fontweight='bold')
# 	ax.ticklabel_format(useOffset=False, style='plain')
# 	plt.xticks(fontsize=22, weight='bold')
# 	plt.yticks(fontsize=22, weight='bold')
# 	plt.show()