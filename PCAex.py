# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 05:47:28 2019

@author: dcamp
"""

from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans 
from sklearn.preprocessing import scale

# Exploratory data analysis (EDA, for short)
plt.style.use('ggplot')
wine = datasets.load_wine()
print(type(wine))
print(wine.keys())
print(type(wine.data), type(wine.target))
print(wine.data.shape)
print(wine.target_names)

X = scale(wine.data)
y = wine.target 

model = PCA()
pca_features1 = model.fit_transform(X)
features = range(len(model.components_))
plt.figure()
plt.bar(features, model.explained_variance_)
plt.xticks(features)
plt.xlabel('feature')
plt.ylabel('variance')

### Making a cut to select the 6 more relevant components (the ones with more variance)
# As an important remark, PCA does not discard features but creates components that encode the provided features
model = PCA(n_components=6)
pca_features = model.fit_transform(X)
features = range(len(model.components_))
plt.figure()
plt.bar(features, model.explained_variance_)
plt.xticks(features)
plt.xlabel('feature')
plt.ylabel('variance')



modelTSNE = TSNE(learning_rate=800)
pca_features_red = modelTSNE.fit_transform(pca_features)
plt.figure()
plt.scatter(pca_features_red[:,0], pca_features_red[:,1], c=y)

modelKmeans = KMeans(n_clusters=3)
modelKmeans.fit(scale(X))
labels = modelKmeans.predict(scale(X))
plt.figure()
plt.scatter(pca_features_red[:,0], pca_features_red[:,1], c=labels)

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y, labels, labels=None, sample_weight=None)
print(conf_matrix)
