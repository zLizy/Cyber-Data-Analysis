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

df = pd.read_csv('transformed_data.csv')

x = df.iloc[: ,0:-2].values
y = df.iloc[:, -1].values

print x.shape,y.shape


x_array = np.array(x)
y_array = np.array(y)

#x_array = x_array[:,0:-2]
#%%

x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size = 0.2)#test_size: proportion of train/test data

#x_train=
# choose the same number of fraud case from training set
fraud_train_y = y_train[y_train==1]
nonfraud_train_y = y_train[y_train==0]
fraud_train_x = x_train[y_train==1]
nonfraud_train_x = x_train[y_train==0]

num_fraud = len(fraud_train_y)
num_non_fraud = len(nonfraud_train_y)

num_test_fraud = len(y_test[y_test==1])

def ran_sample(ds_x,ds_y,total,size):

    idx = list(np.random.randint(total, size=size))
    sample_x = ds_x[idx]
    sample_y = ds_y[idx]

    return sample_x,sample_y

'''fraud data'''
fraud_train_x, fraud_train_y = ran_sample(fraud_train_x,fraud_train_y,num_test_fraud,num_test_fraud)

'''DS 1: proportion 15%'''
#number of non-fraud would be 351
num_non_fraud_ds1 = num_test_fraud/15*85
nonfraud1_train_x, nonfraud1_train_y = ran_sample(nonfraud_train_x,nonfraud_train_y,num_non_fraud,num_non_fraud_ds1)

X_train_ds1 = np.vstack((fraud_train_x,nonfraud1_train_x))
Y_train_ds1 = np.append(fraud_train_y,nonfraud1_train_y)

'''DS 2: proportion 10%'''
#number of non-fraud would be 351
num_non_fraud_ds2 = num_test_fraud/10*90
nonfraud2_train_x, nonfraud2_train_y = ran_sample(nonfraud_train_x,nonfraud_train_y,num_non_fraud,num_non_fraud_ds2)

X_train_ds2 = np.vstack((fraud_train_x,nonfraud2_train_x))
Y_train_ds2 = np.append(fraud_train_y,nonfraud2_train_y)

'''DS 3: proportion 5%'''
#number of non-fraud would be 351
num_non_fraud_ds3 = num_test_fraud/5*95
nonfraud3_train_x, nonfraud3_train_y = ran_sample(nonfraud_train_x,nonfraud_train_y,num_non_fraud,num_non_fraud_ds3)

X_train_ds3 = np.vstack((fraud_train_x,nonfraud3_train_x))
Y_train_ds3 = np.append(fraud_train_y,nonfraud3_train_y)

'''DS 4: proportion 2%'''
#number of non-fraud would be 351
num_non_fraud_ds4 = num_test_fraud/2*98
nonfraud4_train_x, nonfraud4_train_y = ran_sample(nonfraud_train_x,nonfraud_train_y,num_non_fraud,num_non_fraud_ds4)

X_train_ds4 = np.vstack((fraud_train_x,nonfraud4_train_x))
Y_train_ds4 = np.append(fraud_train_y,nonfraud4_train_y)
#%%
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
    #pdb.set_trace()
    '''
    x=np.vstack((x_train,x_test))
    y=np.vstack((y_train,y_test))
    y=np.reshape(y,[len(y),])
    #print x.shape, y.shape
    scores = cross_val_score(clf_r,x,y, cv=10)
    acc=scores.mean()
    #print '10 fold cv '+ str(scores)
    #print 'average precision'+str(acc)
    '''

    #predict_proba = clf_r.predict_proba(x_test)  # the probability of each smple labelled to positive or negative

#    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#    print('\n')

'''ds1 result'''
print "DS1:"
rf_train(X_train_ds1,Y_train_ds1,x_test,y_test)
print("DS2:")
rf_train(X_train_ds2,Y_train_ds2,x_test,y_test)
print("DS3:")
rf_train(X_train_ds3,Y_train_ds3,x_test,y_test)
print("DS4:")
rf_train(X_train_ds4,Y_train_ds4,x_test,y_test)
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
    rf_train(X_res,y_res,x_test,y_test)


    # print("SVM:")
    # svm_train(X_res, y_res, x_test, y_test)
#%%
sm_sample(X_train_ds1,Y_train_ds1,x_test,y_test)
sm_sample(X_train_ds2,Y_train_ds2,x_test,y_test)
sm_sample(X_train_ds3,Y_train_ds3,x_test,y_test)
sm_sample(X_train_ds4,Y_train_ds4,x_test,y_test)
#%%
sm_sample(X_train_ds1[:,0:-2],Y_train_ds1,x_test[:,0:-2],y_test)
sm_sample(X_train_ds2[:,0:-2],Y_train_ds2,x_test[:,0:-2],y_test)
sm_sample(X_train_ds3[:,0:-2],Y_train_ds3,x_test[:,0:-2],y_test)
sm_sample(X_train_ds4[:,0:-2],Y_train_ds4,x_test[:,0:-2],y_test)