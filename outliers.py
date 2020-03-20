import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from minisom import MiniSom

df = pd.read_csv('~/Downloads/aust.csv')

labels = df['Y']
X = df.drop(columns='Y')
X = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, X)

som = MiniSom(15, 15, X.shape[1], sigma=3.5, learning_rate=0.5,
              neighborhood_function='triangle',
              random_seed=10)

som.train_random(X, 3000, verbose=True)

plt.figure(figsize=(8, 8))
wmap = {}
sample = 0
for x, y in zip(X, labels):
    w = som.winner(x)
    wmap[w] = sample
    plt.text(w[0]+.5, w[1]+.5, str(y),
    color = plt.cm.rainbow(y/2), fontdict={'weight': 'bold', 'size': 11})
    sample = sample + 1
plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
plt.grid()
plt.savefig('som_labels_.png')
plt.show()

