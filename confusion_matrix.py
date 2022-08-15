import sklearn.metrics as skm
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, f1_score

# pip install scikit-learn

'''
binary classification
'''

# y_true = ['not_cassava', 'cassava', 'not_cassava', 'not_cassava', 'not_cassava', 'cassava', 'cassava', 'not_cassava', 'not_cassava', 'not_cassava', 'cassava', 'not_cassava']

# y_pred = ['not_cassava', 'not_cassava', 'cassava', 'not_cassava', 'not_cassava', 'not_cassava', 'cassava', 'not_cassava', 'not_cassava', 'cassava', 'not_cassava', 'cassava']
# CM = confusion_matrix(y_true, y_pred, labels=['cassava', 'not_cassava'])
# print(CM)

'''
multiple class classification
'''

y_true = ['maize', 'cassava', 'sugarcrane', 'maize', 'rice', 'cassava', 'cassava', 'sugarcrane', 'rice', 'sugarcrane', 'cassava', 'rice']

y_pred = ['maize', 'rice', 'cassava', 'maize', 'sugarcrane', 'rice', 'cassava', 'sugarcrane', 'rice', 'cassava', 'maize', 'cassava']
CM = multilabel_confusion_matrix(y_true, y_pred, labels=['cassava', 'rice', 'maize', 'sugarcrane'])
# print(CM)
# print(skm.classification_report(y_true,y_pred))

'''
plot confusion matrix
'''

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.figure(figsize=(25, 15))
	fontsize = 25
	plt.rcParams.update({'font.size': fontsize})
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks=np.arange(len(classes))
	plt.xticks(tick_marks,classes, rotation=0, fontsize=fontsize)
	plt.yticks(tick_marks,classes, rotation=0, fontsize=fontsize)
	if normalize:
		cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
		cm=np.around(cm,decimals=2)
		cm[np.isnan(cm)]=0.0
		print('Normalized confusion matrix')
	else:
		print('Confusion matrix, without normalization')
	thresh=cm.max()/2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				horizontalalignment="center", fontsize=fontsize,
				color="white" if cm[i, j] > thresh else "black")
		plt.tight_layout()
		plt.ylabel('True label', fontsize=fontsize)
		plt.xlabel('Predicted label', fontsize=fontsize)
	plt.show()

classes = ['cassava', 'rice', 'maize', 'sugarcrane']
cm_result = confusion_matrix(y_true, y_pred, labels=classes)
plot_confusion_matrix(cm_result, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens)