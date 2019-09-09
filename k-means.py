# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:30:47 2019

@author: damian.campo
"""

from sklearn import datasets
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

#   Load dataset
iris = datasets.load_iris()
samples = iris.data
#print(samples)

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

