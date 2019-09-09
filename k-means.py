# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 22:42:17 2019

@author: damian.campo
"""

from sklearn import datasets
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler # See Normalizer and StandardScaler



#   Load dataset
iris = datasets.load_iris()
samples = iris.data
#print(samples)

#   Preprocess the samples so that all features are between 0 and 1
scaler = MaxAbsScaler()
scaler.fit(samples)
samples = scaler.transform(samples)

model = KMeans(n_clusters=3)
model.fit(samples)
labels = model.predict(samples)

target = iris.target
plt.figure()
#   for plotting purposes only features 2 and 3 are taking into consideration
plt.scatter(samples[:,2], samples[:,3], c=target)
plt.title('Ground truth clusters  (Based on 2 features)')



centroids = model.cluster_centers_
centroids_x = centroids[:,2]
centroids_y = centroids[:,3]

plt.figure()
plt.scatter(samples[:,2], samples[:,3], c=labels, alpha=0.5)
plt.scatter(centroids_x, centroids_y, c='red', marker='+', s=100)
plt.title('Predicted clusters K-means (Based on 2 features)')

###### Evaluate the performance of clusters

print(model.inertia_)
#   Inerto is a measurement of K-means that encodes how tight clusters are. The lower, the tighter (more concentrated they are)
#   Keep in mind that K-means minimizes inertia

k_samples = list(range(1,11))
inertia_samples = []
for k in k_samples:
    model = KMeans(n_clusters=k)
    model.fit(samples)
    inertia_samples.append(model.inertia_)

plt.figure()
plt.plot(k_samples, inertia_samples, c = 'red', marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
# The elbow rule says the optimal n. of clusters is reached when the derivative of inertia w.r.t the number of cluster starts being null

