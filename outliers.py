import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from minisom import MiniSom


# ====================================
# INITIALIZATION
# ====================================
df = pd.read_csv('~/Downloads/aust.csv')
labels = df['Y']
X = df.drop(columns='Y')
X = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, X)
grid_dim = int(X.shape[1]*0.5)
som = MiniSom(grid_dim, grid_dim, X.shape[1], sigma=grid_dim/2, learning_rate=0.1,
              neighborhood_function='gaussian',
              random_seed=10)

X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=123)


# ======================
# TRAIN
# ======================
som.pca_weights_init(X_train)
print("Training...")
som.train_batch(X_train, grid_dim**2*800, verbose=True)  # random training
print("\n...done!")
#som.train_random(X, 7*7*600, verbose=True)


# use different colors and markers for each label
plt.figure(figsize=(grid_dim, grid_dim))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
markers = ['*', 'D']
colors = ['C0', 'C1']
for x, y in zip(X_train, y_train):
    w = som.winner(x)  # getting the winner coordinates
    # place a marker on the winning position for the sample xx
    i = 0 if y == 1 else 1
    plt.plot(w[0]+.5, w[1]+.5, markers[i], markerfacecolor='None',
             markeredgecolor=colors[i], markersize=12, markeredgewidth=2)
plt.axis([0, grid_dim, 0, grid_dim])
plt.colorbar()
plt.savefig('som_labels_train.png')
plt.show()
plt.close()

# train classification plot
plt.figure(figsize=(grid_dim, grid_dim))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
#wmap = {}
sample = 0
wmap = som.labels_map(X, labels)
default_class = np.sum(list(wmap.values())).most_common()[0][0]
for x, y in zip(X_train, y_train):
    w = som.winner(x)
    #wmap[w] = sample
    if w in wmap:
        label = wmap[w].most_common()[0][0]
    else:
        print('plot NON PRESENT')
        label = default_class
    plt.text(w[0]+.5, w[1]+.5, str(label),
    color = plt.cm.rainbow(y/2), fontdict={'weight': 'bold', 'size': 11})
    sample = sample + 1
plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
plt.colorbar()
#plt.grid(linestyle='--', linewidth=.4, which="both")
plt.savefig('som_labels__train_hat.png')
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

# classification report
def classify(som, data, class_assigments):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = class_assignments
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


# ======================
# TEST
# ======================

class_assignments = som.labels_map(X_train, y_train)
y_hat_train = classify(som, X_train, class_assignments)
y_hat = classify(som, X_test, class_assignments)
print("******** Test Classification Report ********")
print(classification_report(y_test, y_hat))

# use different colors and markers for each label TEST
plt.figure(figsize=(grid_dim,grid_dim))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
markers = ['o', 'D']
colors = ['C0', 'C1']
wmap = som.labels_map(X_test, y_test)
for x in X_test:
    w = som.winner(x)  # getting the winner coordinates
    # place a marker on the winning position for the sample xx
    label = wmap[w].most_common()[0][0]
    i = 0 if label == -1 else 1
    plt.plot(w[0]+.5, w[1]+.5, markers[i], markerfacecolor='None',
             markeredgecolor=colors[i], markersize=12, markeredgewidth=2)
plt.axis([0, grid_dim, 0, grid_dim])
plt.colorbar()
plt.savefig('som_labels_test_hat.png')
plt.show()
plt.close()

y_test = y_test.to_list()
y_train = y_train.to_list()

tot_err_train = [0 if y_train[i] == y_hat_train[i] else 1 for i in range(len(y_hat_train))]
tot_err_train = round(np.sum(tot_err_train)/len(y_hat_train), 2)

tot_err_test = [0 if y_test[i] == y_hat[i] else 1 for i in range(len(y_hat))]
tot_err_test = round(np.sum(tot_err_test)/len(y_hat), 2)

print(f"Grid dimension (gaussian) = {(grid_dim, grid_dim)}")
print(f"Classification Train Error: {round(tot_err_train*100, 2)} %")
print(f"Classification Test Error: {round(tot_err_test*100, 2)} %")