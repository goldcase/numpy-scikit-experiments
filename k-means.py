import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# K Means attempts to cluster data by splitting data into groups of equal variance.
# Requires number of clusters to be specified.
# Centroid: mean of cluster.
# Aims to choose centroids that minimize the inertia, or intra-cluster sum of squared distance from the mean.
# Drawbacks
# Note that inertia makes the assumption that clusters are convex and isotropic (identical in all directions).
# Inertia responds poorly to elongated clusters.
# Inertia is not a normalized metric. PCA can reduce the inflation of Euclidean distances that occur with high-dimensional spaces.
# 1. Choose initial centroid, k samples from the dataset.
# 2. Assign each sample to its nearest centroid
# 3. Create new centroids by taking the mean value of all the samples assigned to each previous centroid.
# K means will always converge, but this might be a local minimum, heavily dependent on centroid initialization.
# As such, centroid initialization is done several times.

# In other words, k-means is EM w/small, all-equal diagonal covar matrix.

def get_noise():
    return (np.random.random()-0.5)*20

x_vals_c1 = [25 + get_noise() for i in xrange(50)]
y_vals_c1 = [25 + get_noise() for i in xrange(50)]
x_vals_c2 = [10 + get_noise() for i in xrange(50)]
y_vals_c2 = [10 + get_noise() for i in xrange(50)]

plt.scatter(x_vals_c1, y_vals_c1, color='r')
plt.scatter(x_vals_c2, y_vals_c2, color='b')
plt.show()
