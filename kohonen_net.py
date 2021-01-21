import math
import pickle
import sys

import numpy as np
import pandas as pd

df = pd.read_csv('./data/vk_pers4.csv', delimiter=';')

weights = df.sample(n=5).reset_index()

alpha = 1
delta_alpha = 0.1


def dist(v1, v2):
    return math.sqrt(sum([(i - j) ** 2 for i, j in zip(v1, v2)]))


print(weights)

while alpha > 0:

    for index, row in df.iterrows():
        min_dist = sys.maxsize
        for index_w, weight in weights.iterrows():
            dist_w = dist(row.values[1:], weight.values[2:])
            if dist_w < min_dist:
                min_dist = dist_w
                i = index_w

        weight = weights.iloc[i]
        weight_values = np.array(weight.values[2:])
        row_values = np.array(row.values[1:])
        weight.values[2:] = weight_values + alpha * (row_values - weight_values)

    alpha -= delta_alpha

categories = {}

df_kmeans = set(df.kmeans)

for i in weights.index:
    categories[i] = {}
    for j in df_kmeans:
        categories[i][j] = 0

for index, row in df.iterrows():
    min_dist = sys.maxsize
    for index_w, weight in weights.iterrows():
        dist_w = dist(row.values[1:], weight.values[2:])
        if dist_w < min_dist:
            min_dist = dist_w
            i = index_w

    kmeans_value = df.iloc[index]['kmeans']
    categories[i][kmeans_value] += 1

with open('data.pickle', 'wb') as f:
    pickle.dump(categories, f)

for cluster_num, values in categories.items():
    print(f'Кластер {cluster_num} :\n')
    sum_ = sum(values.values())
    for c, v in values.items():
        print(f'{c} категория: ', (v / sum_) * 100, '%', end='   ')
    print('\n')
