import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.cluster import KMeans

def scatter_plot(lithocolors, x, y, facies, lithofacies):
	fig = plt.figure(figsize=(10, 10))
	cmap = colors.ListedColormap(lithocolors)
	scatter = plt.scatter(x, y, c=facies, s=30, edgecolors='black', alpha=1.0, marker='o', cmap=cmap)
	# plt.xlim(0, 350); plt.ylim(0, 80)
	plt.legend(handles=scatter.legend_elements()[0], labels=lithofacies, frameon=True, loc='lower right')
	plt.show()

def compute_kmeans(x, y, number_of_classes):
	'''
	After finished iteration, this function will locate the distances between centroids (number of classes) and each data point.
    '''
	data = np.zeros(shape=(len(x), 2), dtype=float)
	data[:, 0] = x; data[:, 1] = y
	vector_data = data.reshape(-1, 1) 
	random_centroid = 42 # interger number range 0-42
	kmeans = KMeans(n_clusters = number_of_classes, random_state = random_centroid).fit(vector_data)
	kmeans = kmeans.cluster_centers_[kmeans.labels_]
	kmeans = kmeans.reshape(data.shape)
	return kmeans 

def normalized_data(data, lowest_value, highest_value):
    data = (data - data.min()) / (data.max() - data.min())
    return data * (highest_value - lowest_value) + lowest_value

def find_nearest(data):
	'''
	K-means are unsupervised learning so that the return centroid distances need to group the data points into decided classes.
	'''
	x = normalized_data(data[:, 0], 0, 1)
	unique_x = np.unique(x)
	kmean_data = np.zeros(shape=(len(x), 1), dtype=float)
	# NOTE loop x
	for i in range (0, len(x)):
		difference_array = np.absolute(x[i] - unique_x)
		index = difference_array.argmin()
		kmean_data[i, 0] = index
	return kmean_data