# coding: utf-8
import datetime
import time
import random
import pdb

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from operator import itemgetter
from itertools import groupby
import numpy as np
import pandas as pd
from pandas import DataFrame
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import cross_val_score
import pdb
import crash_on_ipy

df = pd.read_csv('transformed_data.csv')

x = df.iloc[: ,0:-2].values
y = df.iloc[:, -1].values

print x.shape,y.shape


x_array = np.array(x)
y_array = np.array(y)

X = x_array[:,0:-2]
#%%
def sm_sample(x_train,y_train,x_test,y_test):
    sm = SMOTE()
    # x_array = np.array(x)
    # y_array = np.array(y)
    usx = x_train.astype('float64')
    usy = y_train.astype('float64')

    X_res, y_res = sm.fit_sample(usx, usy)
    print(format(Counter(y_train)))
    print('Resampled dataset shape {}'.format(Counter(y_res)))

    #print("Random Forest: ")
    fp_data,fn_data=rf_train(X_res,y_res,x_test,y_test)
    return fp_data,fn_data
    
def rf_train(x_train,y_train,x_test,y_test):
    y_train=np.reshape(y_train,[len(y_train),1])
    y_test=np.reshape(y_test,[len(y_test),1])

    clf_r = RandomForestClassifier(max_depth=2, random_state=0)
    clf_r.fit(x_train, y_train)
    y_predict = clf_r.predict(x_test)
    print("random forest accuracy: ")
    
    print accuracy_score(y_test, y_predict)

    TP, FP, FN, TN = 0, 0, 0, 0
    for i in xrange(len(y_predict)):
        if y_test[i] == 1 and y_predict[i] == 1:
            TP += 1
        if y_test[i] == 0 and y_predict[i] == 1:
            FP += 1
        if y_test[i] == 1 and y_predict[i] == 0:
            FN += 1
        if y_test[i] == 0 and y_predict[i] == 0:
            TN += 1
    #print 'TP: ' + str(TP)
    #print 'FP: ' + str(FP)
    #print 'FN: ' + str(FN)
    #print 'TN: ' + str(TN)
    print confusion_matrix(y_test, y_predict) #watch out the element in confusion matrix
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
    #print('precision: '+str(precision))
    print('recall: ' + str(recall))
 #   pdb.set_trace()
    y_test = np.reshape(y_test,[1,len(y_test)])
    
    diff = (y_predict - y_test)[0]
    fp_index = np.where(diff==1)
    fp_data = x_test[fp_index[0]]
    fn_index = np.where(diff==-1)
    fn_data = x_test[fn_index[0]]
    print type(fn_index)
    return fp_data, fn_data
#%% 10 split dataset
from sklearn.model_selection import KFold
kf=KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X)

i=0
fp_data=np.zeros([5,])
for train_index, test_index in kf.split(X):
    print ("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    fp_data,fn_data=sm_sample(x_train,y_train,x_test,y_test)
    
    
