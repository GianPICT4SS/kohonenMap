import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from minisom import MiniSom

df = pd.read_csv('~/Downloads/aust.csv')

labels = df['Y']
X = df.drop(columns='Y')
X = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, X)

som = MiniSom(7, 7, X.shape[1], sigma=3, learning_rate=0.5,
              neighborhood_function='triangle',
              random_seed=10)


som.pca_weights_init(X)
print("Training...")
som.train_batch(X, 1000, verbose=True)  # random training
print("\n...ready!")
#som.train_random(X, 7*7*600, verbose=True)


# use different colors and markers for each label
plt.figure(figsize=(7,7))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
markers = ['o', 's']
colors = ['C0', 'C1']
for cnt, xx in zip(X, labels):
    w = som.winner(cnt)  # getting the winner
    # palce a marker on the winning position for the sample xx
    i = 0 if xx == 1 else 1
    plt.plot(w[0]+.5, w[1]+.5, markers[i], markerfacecolor='None',
             markeredgecolor=colors[i], markersize=12, markeredgewidth=2)
plt.axis([0, 7, 0, 7])
plt.colorbar()
plt.savefig('som_labels.png')
plt.show()
plt.close()


plt.figure(figsize=(7, 7))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
wmap = {}
sample = 0
for x, y in zip(X, labels):
    w = som.winner(x)
    wmap[w] = sample
    plt.text(w[0]+.5, w[1]+.5, str(y),
    color = plt.cm.rainbow(y/2), fontdict={'weight': 'bold', 'size': 11})
    sample = sample + 1
plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
#plt.grid(linestyle='--', linewidth=.4, which="both")
plt.savefig('som_labels_.png')
plt.show()
plt.close()
#  Quantization Error
max_iter = 10**4
q_error = []
t_error = []
iter_x = []
for i in range(max_iter):
    percent = 100 * (i + 1) / max_iter
    rand_i = np.random.randint(len(X))  # This corresponds to train_random() method.
    som.update(X[rand_i], som.winner(X[rand_i]), i, max_iter)
    if (i + 1) % 100 == 0:
        q_error.append(som.quantization_error(X))
        t_error.append(som.topographic_error(X))
        iter_x.append(i)
        #sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}%')

plt.plot(iter_x, q_error, label='quantization error')
plt.plot(iter_x, t_error, label='topological error')

plt.ylabel('error')
plt.xlabel('iteration index')
plt.grid(linestyle='--', linewidth=.4, which="both")

plt.legend()
plt.savefig('quant_top_error.png')
plt.show()
plt.close()

