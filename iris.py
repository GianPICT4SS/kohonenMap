import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from minisom import MiniSom

from sklearn.metrics import classification_report

# import iris dataset
iris = datasets.load_iris()
data = iris.data
labels = iris.target
# data normalization
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)

grid_dim = 7
som = MiniSom(grid_dim, grid_dim, 4, sigma=3, learning_rate=0.5,
              neighborhood_function='triangle', random_seed=10)

# ==================
# TRAIN
# ==================
som.pca_weights_init(data)
print("Training...")
som.train_batch(data, grid_dim**2*500, verbose=True)  # random training
print("\n...done!")

# =======================================================
# VISUALIZATION
# U-Matrix with distance map as backgroud.
# =======================================================

# use different colors and markers for each label TEST
plt.figure(figsize=(grid_dim,grid_dim))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
# use different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']
for x, y in zip(data, labels):
    w = som.winner(x)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[y], markerfacecolor='None',
             markeredgecolor=colors[y], markersize=12, markeredgewidth=2)
plt.axis([0, grid_dim, 0, grid_dim])
plt.colorbar()
plt.title('Triangle')
plt.savefig('PLOTS/som_iris_triangle.png')
plt.show()
plt.close()


# =================================================================================================================
# ERROR
# The quantization error: average distance between each data vector and its BMU.
# The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
# =================================================================================================================
max_iter = 10**4
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
plt.ylabel('Quantization error')
plt.xlabel('iteration index')
plt.grid(linestyle='--', linewidth=.4, which="both")
plt.title('Triangle')
plt.savefig('PLOTS/quant_error_iris_triangle.png')
plt.show()
plt.close()

plt.plot(iter_x, t_error)
plt.ylabel('Topological error')
plt.xlabel('iteration index')
plt.grid(linestyle='--', linewidth=.4, which="both")
plt.title('Triangle')
plt.savefig('PLOTS/top_error_iris_triangle.png')
plt.show()
plt.close()

# ==================================
# CLASSIFICATION
# ==================================

def classify(som, data, class_assigments):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = class_assigments
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            print('NON PRESENT')
            result.append(default_class)
    return result

class_assigments = som.labels_map(data, labels)
y_hat = classify(som, data, class_assigments)

print("******** Classification Report ********")
print(classification_report(labels, y_hat))

tot_err = [0 if y_hat[i] == labels[i] else 1 for i in range(len(y_hat))]
tot_err = round(np.sum(tot_err)/len(y_hat), 2)

print(f"Grid dimension (Triangle)= {(grid_dim, grid_dim)}")
print(f"Classification Error: {round(tot_err*100, 2)} %")

