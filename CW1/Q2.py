#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import csv
import scipy.io
import time

# In[ ]:


for numBins in [512, 1024, 2048] :

    # File Names
    data_test_filename = 'csvs/test_data_' + str(numBins) +'.csv'
    data_train_filename = 'csvs/train_data_' + str(numBins) +'.csv'

    # Get csv raw data
    data_test_raw = pd.read_csv(data_test_filename, header=None)
    data_train_raw = pd.read_csv(data_train_filename, header=None)
    data_test = data_test_raw.values
    data_train = data_train_raw.values

    # Append class to the raw data
    class_train = np.array([])
    class_test  = np.array([])
    for j in range(10):
        for i in range(15):
            class_test  = np.append(class_test,  j+1)
            class_train = np.append(class_train, j+1)

    idx = np.arange(150)

    # Shuffle train data
    np.random.seed(21)
    np.random.shuffle(idx)
    data_train_shuf = data_train[idx]
    class_train_shuf = class_train[idx]

    # Shuffle test data
    np.random.seed(42)
    np.random.shuffle(idx)
    data_test_shuf = data_test[idx]
    class_test_shuf = class_test[idx]

    for numTrees in [64, 128, 256, 512, 1024, 2048, 4096, 8192] :
        for maxDepth in [4, 8, 16, 32, 64] :
            with open('csvs/bigdata6.csv', 'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                for maxFeatures in range(1, numBins, (numBins//16)) :
                    test_start = time.time()
                    # Build Model
                    clf = RandomForestClassifier(n_estimators=numTrees,
                                 max_depth=maxDepth,
                                 max_features=maxFeatures,
                                 bootstrap=True,
                                 criterion="entropy",
                                 random_state=21,
                                 n_jobs=-1)

                    # Fit Model
                    clf.fit(data_train_shuf, class_train_shuf)

                    # Accuracy
                    predictScore = clf.score(data_test_shuf, class_test_shuf)

                    test_end = time.time()

                    packet = [numBins, numTrees, maxDepth, maxFeatures, predictScore, test_end - test_start]

                    writer.writerow(packet)
                csvFile.close()


# In[ ]:


print('YEEEY')
