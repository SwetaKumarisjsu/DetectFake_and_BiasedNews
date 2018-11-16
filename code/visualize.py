import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
# import enchant

import warnings

warnings.filterwarnings("ignore")

unreliable = 1
reliable = 0

"""
1: unreliable
0: reliable
"""
file_path = 'data2.csv'
data = pd.read_csv(file_path)


def counter(data, label):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                  'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                  'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                  'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                  'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                  "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                  'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                  "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'also',
                  'said', '|', '–', '.', ',']

    def filter_text(text):

        lst = []

        words = str(text).lower().split(' ')

        for word in words:
            if word not in stop_words and word not in ['', "”", '—']:
                lst.append(word)

        return lst

    length = np.shape(data)[0]
    paragraph = ''
    for idx in range(length):
        n_label = data['v1'][idx]
        n_text = data['v2'][idx]

        if n_label == label:
            paragraph += str(n_text) + ' '

    if paragraph != ' ':

        filter_words = filter_text(paragraph)
        return Counter(filter_words).most_common(20)
    else:
        raise Exception("Error")


def plot_words(data, type):
    x = counter(data, type)

    for p in x:
        print(p)
    df = pd.DataFrame(x, columns=['Word', 'Count'])
    df.plot.bar(x='Word')
    if type == reliable:
        plt.title('Reliable')
    else:
        plt.title('Unreliable')
    plt.show()


plot_words(data, unreliable)




"""
count1 = Counter(" ".join(data[data['v1'] == 0]["v2"]).split()).most_common(20)
print(count1)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
count2 = Counter(" ".join(data[data['v1'] == 1]["v2"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})


df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show() """