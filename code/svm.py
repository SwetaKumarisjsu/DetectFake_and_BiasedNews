import numpy as np
import pandas as pd
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('data2.csv').head(2000)

f = feature_extraction.text.CountVectorizer(stop_words='english')
X = f.fit_transform(data["v2"].values.astype('U'))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)



Cs = np.arange(500, 2000, 100) #100000
score_train = np.zeros(len(Cs))
score_test = np.zeros(len(Cs))
recall_test = np.zeros(len(Cs))
precision_test= np.zeros(len(Cs))


def train_svm(C, at_index):
    svc = svm.SVC(C=C)
    svc.fit(X_train, y_train)
    score_train[at_index] = svc.score(X_train, y_train)
    score_test[at_index]= svc.score(X_test, y_test)
    recall_test[at_index] = metrics.recall_score(y_test, svc.predict(X_test))
    precision_test[at_index] = metrics.precision_score(y_test, svc.predict(X_test))


for idx in range(len(Cs)):
    train_svm(Cs[idx], idx)


matrix = np.matrix(np.c_[Cs, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns =
             ['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])

print(models.head(n=10))