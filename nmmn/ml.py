"""
Machine learning methods
=========================

"""

import numpy as np
#from . import misc





def AUCmulti(y_true, y_score):
    """
Computes the area under the ROC curve for multiclass classification models. 
Useful for evaluating the performance of such a model.

Assume `y_true` contains the true labels and `y_score` contains predicted probabilities 
for each class.

:param y_true: 1D array listing the labels
:param y_score: multidimensional array of predicted probabilities

Example: AUC for a classification involving 7 labels and 10 instances.

    # Mock data
    ytrue=np.array([6, 2, 6, 6, 6, 6, 5, 1, 5, 0])
    y_score=np.array([[0.11, 0.04, 0.  , 0.  , 0.03, 0.12, 0.69],
       [0.  , 0.03, 0.76, 0.  , 0.  , 0.01, 0.13],
       [0.05, 0.01, 0.  , 0.  , 0.  , 0.27, 0.63],
       [0.09, 0.01, 0.  , 0.  , 0.  , 0.47, 0.43],
       [0.09, 0.  , 0.01, 0.  , 0.08, 0.51, 0.31],
       [0.03, 0.53, 0.  , 0.  , 0.03, 0.17, 0.21],
       [0.17, 0.07, 0.01, 0.  , 0.03, 0.36, 0.32],
       [0.08, 0.3 , 0.09, 0.  , 0.05, 0.16, 0.26],
       [0.01, 0.01, 0.  , 0.  , 0.01, 0.6 , 0.33],
       [0.  , 0.04, 0.08, 0.01, 0.  , 0.37, 0.41]])

    AUCmulti(ytrue, yscore)
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    # Binarize the labels for a multi-class problem
    y_true = label_binarize(y_true, classes=range(y_score.shape[1]))

    # Compute the AUC for each class
    auc = roc_auc_score(y_true, y_score, multi_class='ovr')

    return auc