# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:48:48 2020

@author: dcamp
"""

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
# Loading a dataset contining 30 features for 569 cases (observations) which can be classified in two groups (targets): 'malignant', 'benign'
# for more information type: print(cancer.DESCR)
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
corrmat = df.corr()

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler_std = StandardScaler()
scaler_min_max = MinMaxScaler()
scaler_std.fit(df)
scaler_min_max.fit(df)
# Scale ("normalize") the data in two different ways 
scaled_data_std = scaler_std.transform(df)
scaled_data_min_max = scaler_min_max.transform(df)

from sklearn.decomposition import PCA
number_dimensions = 3
pca = PCA(n_components=number_dimensions)
fit_std = pca.fit(scaled_data_std)
features_pca_std = fit_std.transform(scaled_data_std)
importance_ori_features_std =  fit_std.components_

fit_min_max = pca.fit(scaled_data_min_max)
features_pca_min_max = fit_min_max.transform(scaled_data_min_max)
importance_ori_features_min_max=  fit_min_max.components_
# plt.matshow(importance_ori_features_std,cmap='viridis')
# plt.matshow(importance_ori_features_min_max,cmap='viridis')

plt.figure()
plt.scatter(features_pca_std[:,0],features_pca_std[:,1], c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.title('Ground truth')

plt.figure()
plt.scatter(features_pca_std[:,0],features_pca_std[:,2], c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Third principle component')
plt.title('Ground truth')

'''
plt.figure()
plt.scatter(features_pca_min_max[:,0],features_pca_min_max[:,1],c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')

plt.figure()
plt.scatter(features_pca_min_max[:,0],features_pca_min_max[:,2],c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Third principle component')
'''
# 
from sklearn.mixture import GaussianMixture
# Ideal case
model = GaussianMixture(n_components=2)
model.fit(features_pca_std)
y_hat_2 = model.predict(features_pca_std)

# Trying different models with different number of clusters (components) in a fully unsupervised case (i.e., without knowing the number of labels)
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(features_pca_std)
          for n in n_components]
# BIC and AIC which calculate evaluate the precision of the model and the complexity of it 
plt.figure()
bic_performances = [m.bic(features_pca_std) for m in models]
aic_performances = [m.aic(features_pca_std) for m in models]

sorted_bic_cases = sorted(range(len(bic_performances)), key=lambda k: bic_performances[k])
sorted_aic_cases = sorted(range(len(aic_performances)), key=lambda k: aic_performances[k])

sorted_bic_cases = np.asarray(sorted_bic_cases) + 1
sorted_aic_cases = np.asarray(sorted_aic_cases) + 1

plt.plot(n_components, [m.bic(features_pca_std) for m in models], label='BIC')
plt.plot(n_components, [m.aic(features_pca_std) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');
print('BIC suggests that the best number of components are: ' + str(sorted_bic_cases[0]) + ', '+ str(sorted_bic_cases[1]) + ' and ' + str(sorted_bic_cases[2]))
print('AIC suggest that the best number of components are: ' + str(sorted_aic_cases[0]) + ', '+ str(sorted_aic_cases[1]) + ' and ' + str(sorted_aic_cases[2]))
# By the plots, we should follow the BIC measurement since it actually presented an infection point around 2 and 4 clusters (best options)

# Ideal case based on the BIC measurement 
model = GaussianMixture(n_components=3)
model.fit(features_pca_std)
y_hat_3 = model.predict(features_pca_std)
freq_comp = list()
for n in range(max(y_hat_3) +1):
    freq_comp.append(sum(y_hat_3==n))

sorted_freq_comp = sorted(range(len(freq_comp)), key=lambda k: freq_comp[k])
# Select two clusters that capture more data and redefine the GT 
selec_comp = np.array([sorted_freq_comp[-1], sorted_freq_comp[-2]]) 
ground_T = cancer['target']
modified_GT = np.concatenate((ground_T[y_hat_3==selec_comp[0]], ground_T[y_hat_3==selec_comp[1]]), axis=0)
modified_features_pca_std = np.concatenate((features_pca_std[y_hat_3==selec_comp[0],:], features_pca_std[y_hat_3==selec_comp[1],:]), axis=0)
y_hat_3 = np.concatenate((y_hat_3[y_hat_3==selec_comp[0]], y_hat_3[y_hat_3==selec_comp[1]]), axis=0)

y_hat_3[y_hat_3==selec_comp[0]] = 0
y_hat_3[y_hat_3==selec_comp[1]] = 1

# Measuring performances
from sklearn.metrics import confusion_matrix
# Ideal case
conf_mat = confusion_matrix(ground_T, y_hat_2)
if conf_mat.trace()/np.sum(conf_mat) < 0.5:
    # some relabling to make outputs from the algorithm comparable to the gound truth 
    y_hat_2[y_hat_2 == 1] = 2
    y_hat_2[y_hat_2 == 0] = 1
    y_hat_2[y_hat_2 == 2] = 0
    conf_mat = confusion_matrix(ground_T, y_hat_2)

print('performance of ideal classifier: ' + str(conf_mat.trace()/np.sum(conf_mat)))
print('Over 100%  of data')
# Fully unsupervised case
conf_mat = confusion_matrix(modified_GT, y_hat_3)
if conf_mat.trace()/np.sum(conf_mat) < 0.5:
    # some relabling to make outputs from the algorithm comparable to the gound truth 
    y_hat_3[y_hat_3 == 1] = 2
    y_hat_3[y_hat_3 == 0] = 1
    y_hat_3[y_hat_3 == 2] = 0
    conf_mat = confusion_matrix(modified_GT, y_hat_3)

print('performance of unsupervised classifier: ' + str(conf_mat.trace()/np.sum(conf_mat)))
print('Over ' +  str(len(modified_GT)/len(ground_T) *100) + '%' + ' of data')

plt.figure()
plt.scatter(features_pca_std[:,0],features_pca_std[:,1], c=y_hat_2)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.title('Model based on 2 classes (clusters)')

plt.figure()
plt.scatter(modified_features_pca_std[:,0],modified_features_pca_std[:,1], c=y_hat_3)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.title('Model based on 3 classes (clusters)')






