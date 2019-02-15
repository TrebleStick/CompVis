#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import csv
import scipy.io
import matplotlib.pyplot as plt
import time

# In[2]:


# Load SIFT data
mat_te = scipy.io.loadmat('csvs/desc_te.mat')
mat_tr = scipy.io.loadmat('csvs/desc_tr.mat')

# Access arrays
desc_tr_raw = mat_tr['desc_tr']
desc_te_raw = mat_te['desc_te']

del mat_te
del mat_tr

# Reshape to column
desc_tr = desc_tr_raw.reshape(150)
desc_te = desc_te_raw.reshape(150)

del desc_tr_raw
del desc_te_raw

# In[3]:
# Reshape data
data_train_patches  = desc_tr[0].T
class_train_patches = np.asarray([1] * desc_tr[0].T.shape[0])
data_test_patches  = desc_te[0].T
class_test_patches = np.asarray([1] * desc_te[0].T.shape[0])

# Create column of SIFT patches
for x in range(1,150):
    data_train_patches   = np.concatenate([data_train_patches, desc_tr[x].T])
    class_train_patches  = np.concatenate([class_train_patches, np.asarray([x%10 +1] * desc_tr[x].T.shape[0])])
    data_test_patches   = np.concatenate([data_test_patches, desc_te[x].T])
    class_test_patches  = np.concatenate([class_test_patches, np.asarray([x%10 +1] * desc_te[x].T.shape[0])])
# In[4]:


# Shuffle sift patches
data_train_shuf = data_train_patches
data_test_shuf = data_test_patches
class_train_shuf = class_train_patches
class_test_shuf = class_test_patches

del data_train_patches
del data_test_patches
del class_train_patches
del class_test_patches


# In[6]:
for num_trees in [5,10]:
    for max_depth in [6,9]:
        # codebook_filename = 'csvs/RF_codebook_train_' + str(num_trees) + '_' + str(max_depth) + '.csv'
        # with open(codebook_filename, 'a', newline='') as csvFile:
            # writer = csv.writer(csvFile)
        print('num_trees:, max_depth:', num_trees, max_depth)

        # Build Model
        clf = RandomForestClassifier(n_estimators=num_trees,
                                     max_depth=max_depth,
                                     max_features='auto',
                                     bootstrap=True,
                                     criterion="entropy",
                                     random_state=21,
                                     n_jobs=-1)

        # Fit Model
        clf.fit(data_train_shuf, class_train_shuf)

        # Accuracy
        predictScore = clf.score(data_test_shuf, class_test_shuf)
        print(predictScore)

        # Create transformer
        rf_enc = OneHotEncoder()
        rf_enc.fit(clf.apply(data_train_shuf))

        # del class_test_shuf
        # del class_train_shuf

        # Transform data
        data_train_transf = list([])
        data_test_transf = list([])
        for i in range(150):
            data_train_transf.append(rf_enc.transform(clf.apply(desc_tr[i].T)).toarray())
            data_test_transf.append(rf_enc.transform(clf.apply(desc_te[i].T)).toarray())
        del clf




        print('GOT EEM')

        # Condense to histograms
        data_train = []
        data_test = []

        for i in range(150):
            data_train_temp = data_train_transf[i][0]
            data_test_temp = data_test_transf[i][0]

            for j in range(1, data_train_transf[i].shape[0]):
                data_train_temp += data_train_transf[i][j]
            data_train.append(data_train_temp)
            for j in range(1, data_test_transf[i].shape[0]):
                data_test_temp += data_test_transf[i][j]
            data_test.append(data_test_temp)

        data_train_filename = 'csvs/RF_train_' + str(num_trees) + '_' + str(max_depth) + '.csv'
        data_test_filename = 'csvs/RF_test_' + str(num_trees) + '_' + str(max_depth) + '.csv'

        with open(data_train_filename, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for i in data_train:
                writer.writerow(i)
        csvFile.close()
        with open(data_test_filename, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for i in data_test:
                writer.writerow(i)
        csvFile.close()

        print('Done', num_trees, max_depth)
