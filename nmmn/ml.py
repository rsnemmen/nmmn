"""
Machine learning methods
=========================

"""

import numpy as np
#from . import misc





def scatterfit(x,y,a=None,b=None):
	"""
Compute the mean deviation of the data about the linear model given if A,B
(*y=ax+b*) provided as arguments. Otherwise, compute the mean deviation about 
the best-fit line.

:param x,y: assumed to be Numpy arrays. 
:param a,b: scalars.
:rtype: float sd with the mean deviation.
	"""

	if a==None:	
		# Performs linear regression
		a, b, r, p, err = scipy.stats.linregress(x,y)
	
	# Std. deviation of an individual measurement (Bevington, eq. 6.15)
	N=np.size(x)
	sd=1./(N-2.)* np.sum((y-a*x-b)**2)
	sd=np.sqrt(sd)
	
	return sd
	


def AUCmulti(y_true, y_score):
    """
Computes the area under the ROC curve
   Assume y_true contains the true labels and y_score contains predicted probabilities for each class

    y_true: 1D array listing the labels
    y_score: multidimensional array of probabilities
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    # Binarize the labels for a multi-class problem
    y_true = label_binarize(y_true, classes=range(y_score.shape[1]))

    # Compute the AUC for each class
    auc = roc_auc_score(y_true, y_score, multi_class='ovr')

    return auc