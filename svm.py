
# coding: utf-8

# In[8]:


"""
The SVM model
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os
from DataPreProcessing import doc2Vector

def SVM():
    # Read the data
    if not os.path.isfile('./xtraining.npy') or not os.path.isfile('./xtesting.npy') or not os.path.isfile('./ytraining.npy') or not os.path.isfile('./ytesting.npy'):
        xtrain,xtest,ytrain,ytest = doc2Vector("data/train.csv")
        np.save('./xtraining', xtrain)
        np.save('./xtesting', xtest)
        np.save('./ytraining', ytrain)
        np.save('./ytesting', ytest)
    xtrain = np.load('./xtraining.npy')
    xtest = np.load('./xtesting.npy')
    ytrain = np.load('./ytraining.npy')
    ytest = np.load('./ytesting.npy')

    # Use the built-in SVM for classification
    clf = SVC()
    clf.fit(xtrain, ytrain)
    y_pred = clf.predict(xtest)
    m = ytest.shape[0]
    n = (ytest != y_pred).sum()
    print("SVM model accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%

    # Draw the confusion matrix
    skplt.plot_confusion_matrix(ytest, y_pred)
    plt.show()


# In[9]:


SVM()

