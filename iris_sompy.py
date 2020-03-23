import numpy as np
from sompy.sompy import SOMFactory
from sklearn import datasets

# import iris dataset
iris = datasets.load_iris()
data = iris.data
labels = iris.target



# initialization SOM
sm = SOMFactory().build(data, normalization='var', initialization='pca')
sm.train(n_job=1, verbose=True, train_rough_len=2, train_finetune_len=5)


# The quantization error: average distance between each data vector and its BMU.
# The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))

# component planes view
from sompy.visualization.mapview import View2D
view2D = View2D(10,10,"rand data",text_size=12)
view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True)

# U-matrix plot
from sompy.visualization.umatrix import UMatrixView

umat = UMatrixView(width=10,height=10,title='U-matrix')
umat.show(sm)
