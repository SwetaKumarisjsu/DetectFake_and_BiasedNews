import pandas as pd
import numpy as np

file_path = "../unclean-data/all/train.csv"

LABEL = 'label'
TEXT = 'text'
reader = pd.read_csv(file_path)

data = []

length = np.shape(reader)[0]

# print(length//3); exit(0)

# print(reader['text'][0])

for idx in range(length//3):
    n_label = reader[LABEL][idx]
    n_text = reader[TEXT][idx]

    data.append([n_label, n_text])

def save(data):
    df = pd.DataFrame(data=data, columns=['v1', 'v2'])
    df.to_csv('data2.csv')

save(data)

