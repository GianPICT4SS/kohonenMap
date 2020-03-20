import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from minisom import MiniSom

# import iris dataset
iris = datasets.load_iris()
data = iris.data

som = MiniSom(7, 7, 4, sigma=3, learning_rate=0.5,
              neighborhood_function='triangle', random_seed=10)
som.pca_weights_init(data)
"""
max_iter = 10000
q_error = []
t_error = []
iter_x = []
for i in range(max_iter):
    percent = 100 * (i + 1) / max_iter
    rand_i = np.random.randint(len(data))  # This corresponds to train_random() method.
    som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)
    if (i + 1) % 100 == 0:
        q_error.append(som.quantization_error(data))
        t_error.append(som.topographic_error(data))
        iter_x.append(i)
        #sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}%')

plt.plot(iter_x, q_error)
plt.ylabel('quantization error')
plt.xlabel('iteration index')

"""
