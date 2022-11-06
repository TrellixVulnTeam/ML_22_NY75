import matplotlib.pyplot as plt
import numpy as np
import rasterio
#-----------------------------------------------------------------------------------------#
from numpy import save
from rasterio import features
from rasterio.features import rasterize
from skimage.transform import resize
from skimage import exposure 
from sklearn.preprocessing import RobustScaler
#-----------------------------------------------------------------------------------------#

def ellipse(b, x, a):
    """
	creating simple ellipse function
    """
    p1 = pow(x, 2)/pow(a, 2)
    p2 = np.sqrt(1000 - p1)
    y = b*p2
    return y

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

def plot_band(image, color, min_num, max_num, save_name):
	'''
	General plot
	'''
	plt.imshow(image, cmap=color, vmin=min_num, vmax=max_num)
	plt.axis('off')
	# plt.savefig('pictures/' + save_name + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
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
	# image = image[image <= 1]
	x = image[~np.isnan(image)]
	plt.hist(x.flatten(), bins=30, color='g', histtype='bar', ec='black')
	plt.show()

def normalized_data(y, lowest_value, highest_value):
	y = (y - y.min()) / (y.max() - y.min())
	return y * (highest_value - lowest_value) + lowest_value

def composite_bands(a, b, c, clip):
	'''
	composite bands
	'''
	# print(a.shape, b.shape, c.shape)
	a = normalized_data(a, 0, 255); b = normalized_data(b, 0, 255); c = normalized_data(c, 0, 255)
	# band_stacking = np.stack((a, b, c), axis=2)
	band_stacking = np.stack((a, b, c), axis=2) # red always first
	pLow, pHigh = np.percentile(band_stacking[~np.isnan(band_stacking)], (clip, 100-clip))
	band_stacking = exposure.rescale_intensity(band_stacking, in_range=(pLow, pHigh))  # type: ignore
	# plt.figure(figsize=(20, 20))
	plt.imshow(band_stacking, cmap='viridis', alpha=1.)
	plt.axis('off')
	# plt.savefig('pictures/geo' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def composite_2(a, b, c, clip):
	a = normalized_data(a, 0, 255); b = normalized_data(b, 0, 255); c = normalized_data(c, 0, 255)
	band_stacking = np.stack((a, b, c), axis=2) # red always first
	pLow, pHigh = np.percentile(band_stacking[~np.isnan(band_stacking)], (clip, 100-clip))
	stacked_bands = exposure.rescale_intensity(band_stacking, in_range=(pLow, pHigh))  
	return(stacked_bands)

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

# NOTE replace existing data
def normalization_2(data, drop_cols, log_names):# shift mean to zero
	dummy = data.drop(drop_cols, axis=1)
	scaler = preprocessing.StandardScaler().fit(dummy)
	dummy = scaler.transform(dummy)
	for count, item in enumerate(log_names):
		data[item] = dummy[:, count]
	return data

def remove_outliers(data, log, min_o, max_o):
	q_low = data[log].quantile(min_o)
	q_hi  = data[log].quantile(max_o)
	return data[(data[log] < q_hi) & (data[log] > q_low)]

def hstack_farm(farm_id, crop_type_number):
	band_number = farm_id[farm_id != 0.].reshape(-1, 1)
	vector_length = len(band_number)
	crop_type = np.repeat(crop_type_number, vector_length).reshape(-1, 1)
	band_median = np.repeat(np.median(band_number), vector_length).reshape(-1, 1)
	band_mean = np.repeat(np.mean(band_number), vector_length).reshape(-1, 1)
	return np.hstack((crop_type, band_number, band_median, band_mean))

def hstack_farm_2(farm_id, crop_type_number):
	'''
	compute one farm and then hstack
	'''
	band_number = farm_id[farm_id != 0.].reshape(-1, 1)
	vector_length = len(band_number)
	crop_type = np.repeat(crop_type_number, vector_length).reshape(-1, 1)
	return np.hstack((crop_type, band_number))

def save2npy(date, suffix_, band_array):
	save_file_name = '../datasets/save_npy_2/' + date + suffix_
	print('saving: ', save_file_name)
	save(save_file_name, band_array)

def access_bands(file_, date):
	'''
	B2, B3, B4, B8, veg 
	'''
	B2  = file_ + 'B02.jp2'
	B3  = file_ + 'B03.jp2'
	B4  = file_ + 'B04.jp2'
	B8  = file_ + 'B08.jp2'
	bands = [B2, B3, B4, B8]
	for index, i in enumerate(bands):
		# print('accessing band: ', i)
		band = rasterio.open(i).read(1)
		if index == 0:
			B2 = band
			save2npy(date, '_B02.npy', B2)
		elif index == 1:
			B3 = band
			save2npy(date, '_B03.npy', B3)
		elif index == 2:
			B4 = band
			save2npy(date, '_B04.npy', B4)
		elif index == 3:
			B8 = band
			save2npy(date, '_B08.npy', B8)
	veg = NDVI(B8, B4)
	save2npy(date, '_veg.npy', veg)

def normalize_clip(nor_data):
	ROW, COL = nor_data.shape
	nor_data = nor_data.reshape(-1, 1)
	scaler = RobustScaler()
	nor_data = scaler.fit_transform(nor_data)
	nor_data = nor_data.reshape(ROW, COL)
	min_num, max_num = clip(nor_data, 95)
	for i in range (0, ROW):
		for ii in range (0, COL):
			if nor_data[i, ii] < min_num or nor_data[i, ii] > max_num:
				nor_data[i ,ii] = 0.001 # avoid 0. --> otherwise zero will be clipped during reshape
			elif nor_data[i, ii] == 0.:
				nor_data[i ,ii] = 0.001 # avoid 0. --> otherwise zero will be clipped during reshape
	return nor_data

def MAE(data_y, model):
	sum = 0.
	for i in range (0, len(data_y)):
		sum += abs(data_y[i] - model[i])
	return sum/len(data_y)

def MSE(data_y, model):
	sum = 0.
	for i in range (0, len(data_y)):
		sum += (data_y[i] - model[i])**2
	return sum/len(data_y)

def linear_equation(slope, weights, intercept):
	return slope*weights + intercept

# https://gist.github.com/sagarmainkar/41d135a04d7d3bc4098f0664fe20cf3c
def  cal_cost(theta, X, y):
    '''
    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas 
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))
    where:
        j is the no of features
    '''
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):
        prediction = np.dot(X,theta)
        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
        theta_history[it,:] =theta.T
        cost_history[it]  = cal_cost(theta, X, y)
    return theta, cost_history, theta_history
