
# coding: utf-8

# In[17]:


"""
The naive bayes model
"""
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt
from DataPreProcessing import doc2Vector

def NaiveBayes():
    # Reading the data
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

    # Use the built-in Naive Bayes classifier
    gnb = GaussianNB()
    gnb.fit(xtrain,ytrain)
    y_pred = gnb.predict(xtest)
    m = ytest.shape[0]
    n = (ytest != y_pred).sum()
    print("Naive Bayes model accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 72.94%

    # Draw the confusion matrix
    skplt.plot_confusion_matrix(ytest, y_pred)
    plt.show()


# In[18]:




