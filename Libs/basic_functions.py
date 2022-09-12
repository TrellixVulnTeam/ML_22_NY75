import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from sklearn.cluster import KMeans
from bokeh.palettes import Set1

def scatter_plot(lithocolors, x, y, facies, lithofacies):
	fig = plt.figure(figsize=(10, 10))
	cmap = colors.ListedColormap(lithocolors)
	scatter = plt.scatter(x, y, c=facies, s=30, edgecolors='black', alpha=1.0, marker='o', cmap=cmap)
	# plt.xlim(0, 350); plt.ylim(0, 80)
	plt.legend(handles=scatter.legend_elements()[0], labels=lithofacies, frameon=True, loc='lower right')
	plt.show()

def scatter_plot_2(y):
	plt.figure(figsize=(10, 10))
	x = np.linspace(0, len(y), len(y))
	scatter = plt.scatter(x, y, color='orangered', s=50, edgecolors='black', alpha=1.0, marker='o')
	plt.title('Input Data', fontweight='bold')
	plt.xlabel('index'); plt.ylabel('intensity')
	plt.savefig('../drawing/image_out/' + 'beforeNor' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def KDE_plot(input_data):
	fig, ax = plt.subplots()
	sns.kdeplot(data=input_data[:, 0], color='#27AE60', shade=True, multiple='stack', ax=ax, label='RobustScaler')
	sns.kdeplot(data=input_data[:, 1], color='#D4AC0D', shade=True, multiple='stack', ax=ax, label='MinMaxScaler')
	plt.title('Kernel Density Estimation (KDE)', fontweight='bold')
	plt.xlabel('intensity'); plt.ylabel('density')
	plt.legend(loc='upper left')
	plt.savefig('../drawing/image_out/' + 'KDE_nor' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
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

def visualize_classifier(model, lithocolors, x, y, lithofacies, labels, title, ylabel, xlabel):
	'''
	For fancy colors please intall: 
	pip install bokeh
	'''
	# TODO scatter plot
	# _, ax = plt.rcParams['figure.figsize'] = (12, 8)
	fig = plt.figure(figsize=(10, 10))
	scatter_colors = colors.ListedColormap(lithocolors)
	scatter = plt.scatter(x, y, c=labels, s=30, cmap=scatter_colors, zorder=3, edgecolors='black', alpha=0.9)
	plt.title(title); plt.ylabel(ylabel); plt.xlabel(xlabel)
	plt.xlim(30, 140)
	plt.legend(handles=scatter.legend_elements()[0], labels=lithofacies, frameon=True, framealpha=1.0, loc='upper right')

	# TODO decsion trees	
	X = np.zeros(shape=(len(x), 2), dtype=float)
	X[:, 0] = x; X[:, 1] = y
	model.fit(X, labels)
	xx, yy = np.meshgrid(np.linspace(30, 140, 200), np.linspace(y.min(), y.max(), 200))
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	n_classes = len(np.unique(labels))
	cmap = colors.ListedColormap(Set1[9][0:9]) # use color pallete from bokeh
	plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, zorder=1)
	plt.show()