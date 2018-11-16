import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn import svm

def read_csv(filename):
    csv_file = open(filename, mode='r')
    csv_reader = csv.DictReader(csv_file)
    return csv_reader

def fetch_csv(csv_reader, limit):
    lst = []

    count = 0

    for row in csv_reader:
        lst.append(row)
        count += 1
        if count == limit:
            return lst


def pretty_csv_row(csv_reader, csv_row):
    keys = csv_reader.fieldnames

    s = ''

    for idx in range(len(keys) - 1):
        s += csv_row[keys[idx]] + ','

    s += csv_row[keys[len(keys) - 1]]
    return s


def extra_data(reader, n):
    return parse_rows(reader, n)

def parse_rows(csv_reader, n):
    X = []
    y = []

    c = 0
    for row in csv_reader:
        X.append(row['text'])
        y.append(int(row['label']))
        print(row['label'])

        if c == n:
            break
    return X, y

file_path = "../unclean-data/all/train.csv"


reader = pd.read_csv(file_path).head(1000)

f = feature_extraction.text.CountVectorizer(stop_words='english')


X = f.fit_transform(reader["text"].values.astype('U'))
y = reader['label']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)


svc = svm.SVC(C=10000)
svc.fit(X_train, y_train)

print(svc.score(X_test, y_test))