import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## set random seed
seed = 726
## read data
adult = pd.read_csv('wholesale_customers.csv')
adult = adult.drop(columns=['Channel','Region'])
cols = adult.columns

# Question 1
mean = adult.mean(axis=0)
min = adult.min(axis=0)
max = adult.max(axis=0)

print '\n Answer to Question 1: '
for i in range(len(mean)):
    print '  The mean of %s : %.2f'%(cols[i],mean[i])
    print '  The range of %s : [%.2f, %.2f]'%(cols[i],min[i],max[i])

# Question 2
print '\n Answer to Question 2: '
x = []
title = []
for i in range(adult.shape[1]):
    for j in range(i+1,adult.shape[1]):
        x.append([adult.iloc[:,i],adult.iloc[:,j]])
        title.append([cols[i],cols[j]])
x = np.array(x)
x = x.reshape(x.shape[0],x.shape[2],x.shape[1])
plt.figure(figsize=(15,9))
for k in range(x.shape[0]):
    kmeans = KMeans(n_clusters=3, random_state=seed).fit(x[k])
    plt.subplot(3,5,k+1)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(x[k,:,0], x[k,:,1], c=kmeans.labels_)
    plt.title(title[k])
plt.savefig('kmeans_15.png')
print '  Picture saved as kmeans_15.png.'

# Question 3
k_list = [3,5,10]
BC = np.zeros(len(k_list))
WC = np.zeros(len(k_list))
for i in range(len(k_list)):
    K = k_list[i]
    kmeansi = KMeans(n_clusters=K,random_state=seed).fit(adult)
    for j in range(K):
        for l in range(j+1,K):
            for k in range(adult.shape[1]):
                BC[i] += np.square(kmeansi.cluster_centers_[j][k]-kmeansi.cluster_centers_[l][k])
    WC[i] = kmeansi.inertia_

print '\n Answer to Question 3: '
for i in range(len(k_list)):
    print '  When k=%d: BC=%f WC=%f BC/WC=%f.'% (k_list[i],BC[i],WC[i],BC[i]/WC[i])
