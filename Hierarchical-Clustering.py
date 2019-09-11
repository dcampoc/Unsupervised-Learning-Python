# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:01:45 2019

@author: dcamp
"""

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import os
#import numpy as np
os.chdir(r'C:\Users\dcamp\Documents\python-practice\Unsupervised-learning')

df = pd.read_csv('eurovision-2016.csv')
samples = df.iloc[:,2:4].values
samples= samples[:40,:]
# method='complete' indicates that clusters are fully linked (agglomerated) at the end of the algorithm
mergins = linkage(samples, method='complete')
country_names = df.iloc[:,1].values
country_names = country_names[:40]
dendrogram(mergins,labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()
#   The y-axis in the dendrogram encodes the distance between clusters

#   Intermidiate clusters
from scipy.cluster.hierarchy import fcluster
#   Clusters with distance 7 or less merged together as a single cluster  
labels = fcluster(mergins, 7, criterion='distance')
pairs = pd.DataFrame({'labels':labels, 
                      'countries':country_names})
print(pairs)