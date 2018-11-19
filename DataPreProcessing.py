
# coding: utf-8

# In[ ]:


import re
import string
import os
import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords
import warnings
from gensim import utils

#Adding this to filter all the warning messages
warnings.simplefilter("ignore")

def dataClean(textdata):
    """
    Removing special characters
    Removing empty cells
    Removing stopwords
    """
    textdata = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", textdata)
    textdata = textdata.lower().split()
    stops = set(stopwords.words("english"))
    textdata = [w for w in textdata if not w in stops]
    textdata = " ".join(textdata)
    return (textdata)


def dataCleanup(textdata):
    textdata = dataClean(textdata)
    textdata = textdata.translate(str.maketrans("", "", string.punctuation))
    return textdata


def labeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences


# In[ ]:



def doc2Vector(path,vector_dimension=300):
    """
    Generating training and testing data using Doc2Vec
    """
    data = pd.read_csv(path)

    missing_rows = []
    
    for i in range(len(data)):
        if data.loc[i, 'text'] != data.loc[i, 'text']:
            missing_rows.append(i)
            
    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    for i in range(len(data)):
        data.loc[i, 'text'] = dataCleanup(data.loc[i,'text'])

    x = labeledSentences(data['text'])
    y = data['label'].values

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,
                         seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.iter)

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]

    j = 0
    
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels


# In[ ]:


def clean_data():
    """
    Generating processed string
    """
    path = 'data/train.csv'
    vector_dimension=300

    data = pd.read_csv(path)

    missing_rows = []
    
    for i in range(len(data)):
        if data.loc[i, 'text'] != data.loc[i, 'text']:
            missing_rows.append(i)
            
    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    for i in range(len(data)):
        data.loc[i, 'text'] = dataCleanup(data.loc[i,'text'])

    data = data.sample(frac=1).reset_index(drop=True)

    x = data.loc[:,'text'].values
    y = data.loc[:,'label'].values
    print(x,y)

    train_size = int(0.8 * len(y))
    test_size = len(x) - train_size
    
    print("train_size: ",train_size)
    print("test_size", test_size)

    xtrain = x[:train_size]
    xtest = x[train_size:]
    ytrain = y[:train_size]
    ytest = y[train_size:]

    np.save('x_training_data_shuffled.npy',xtrain)
    np.save('x_testing_data_shuffled.npy',xtest)
    np.save('y_training_data_shuffled.npy',ytrain)
    np.save('y_testing_data_shuffled.npy',ytest)
    

