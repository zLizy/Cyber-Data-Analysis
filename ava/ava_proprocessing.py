# coding: utf-8
import datetime
import time
import random

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import cross_val_score
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

def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)
def currency_trans(currencycode, amount):
    if currencycode == 'MXN':
        amount = 0.043*amount
    if currencycode =='AUD':
        amount = 0.63*amount
    if currencycode =='NZD':
        amount = 0.59*amount
    if currencycode == 'GBP':
        amount = 1.13*amount
    if currencycode == 'SEK':
        amount = 0.095*amount
    return amount

if __name__ == "__main__":
    src = '/home/ava/Dropbox/CyberData/ass1/data_for_student_case.csv'
    ah = open(src, 'r')
    x = []#contains features
    y = []#contains labels
    data = []
    color = []
    (issuercountry_set, txvariantcode_set, currencycode_set, shoppercountry_set, interaction_set,
    verification_set, accountcode_set, mail_id_set, ip_id_set, card_id_set) = [set() for _ in xrange(10)]
    (issuercountry_dict, txvariantcode_dict, currencycode_dict, shoppercountry_dict, interaction_dict,
    verification_dict, accountcode_dict, mail_id_dict, ip_id_dict, card_id_dict) = [{} for _ in xrange(10)]
    #label_set
    #cvcresponse_set = set()
    ah.readline()#skip first line
    for line_ah in ah:
        if line_ah.strip().split(',')[9]=='Refused':# remove the row with 'refused' label, since it's uncertain about fraud
            continue
        if 'na' in str(line_ah.strip().split(',')[14]).lower() or 'na' in str(line_ah.strip().split(',')[4].lower()):
            continue
        bookingdate = string_to_timestamp(line_ah.strip().split(',')[1])# date reported flaud
        issuercountry = line_ah.strip().split(',')[2]#country code
        issuercountry_set.add(issuercountry)
        txvariantcode = line_ah.strip().split(',')[3]#type of card: visa/master
        txvariantcode_set.add(txvariantcode)
        issuer_id = float(line_ah.strip().split(',')[4])#bin card issuer identifier
        currencycode = line_ah.strip().split(',')[6]
        amount = float(line_ah.strip().split(',')[5])#transaction amount in minor units
        amount = currency_trans(currencycode, amount)
        currencycode_set.add(currencycode)
        shoppercountry = line_ah.strip().split(',')[7]#country code
        shoppercountry_set.add(shoppercountry)
        interaction = line_ah.strip().split(',')[8]#online transaction or subscription
        interaction_set.add(interaction)
        if line_ah.strip().split(',')[9] == 'Chargeback':
            label = 1#label fraud
        else:
            label = 0#label save
        verification = line_ah.strip().split(',')[10]#shopper provide CVC code or not
        verification_set.add(verification)
        cvcresponse = line_ah.strip().split(',')[11]#0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
        #if cvcresponse > 2:
        #    cvcresponse = 3
        year_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%Y-%m-%d %H:%M:%S').year
        month_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%Y-%m-%d %H:%M:%S').month
        day_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%Y-%m-%d %H:%M:%S').day
        creationdate = str(year_info)+'-'+str(month_info)+'-'+str(day_info)#Date of transaction 
        creationdate_stamp = string_to_timestamp(line_ah.strip().split(',')[12])#Date of transaction-time stamp
        accountcode = line_ah.strip().split(',')[13]#merchant’s webshop
        accountcode_set.add(accountcode)
        mail_id = int(float(line_ah.strip().split(',')[14].replace('email','')))#mail
        mail_id_set.add(mail_id)
        ip_id = int(float(line_ah.strip().split(',')[15].replace('ip','')))#ip
        ip_id_set.add(ip_id)
        card_id = int(float(line_ah.strip().split(',')[16].replace('card','')))#card
        card_id_set.add(card_id)
        data.append([issuercountry, txvariantcode, issuer_id, amount, currencycode,
                    shoppercountry, interaction, verification, cvcresponse, creationdate_stamp,
                     accountcode, mail_id, ip_id, card_id, label, creationdate])# add the interested features here
        #y.append(label)# add the labels
    raw_data = sorted(data, key = lambda k: k[-1])
#%%    
data=DataFrame(raw_data)
names=['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode',
                    'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp',
                     'accountcode', 'mail_id', 'ip_id', 'card_id', 'label', 'creationdate']
data.columns= names
del data['currencycode']
del data['creationdate']
print data.head()
#%%
x = []#contains features
y = []#contains labels
l_data=np.array(data).tolist()
for item in l_data:#split data into x,y
    #print item
    x.append(item[0:-1])
    y.append(item[-1])

X = x[:]
#%%
'''map number to each categorial feature'''
for item in list(issuercountry_set):
    issuercountry_dict[item] = list(issuercountry_set).index(item)
for item in list(txvariantcode_set):
    txvariantcode_dict[item] = list(txvariantcode_set).index(item)
#for item in list(currencycode_set):
#    currencycode_dict[item] = list(currencycode_set).index(item)
for item in list(shoppercountry_set):
    shoppercountry_dict[item] = list(shoppercountry_set).index(item)
for item in list(interaction_set):
    interaction_dict[item] = list(interaction_set).index(item)
for item in list(verification_set):
    verification_dict[item] = list(verification_set).index(item)
for item in list(accountcode_set):
    accountcode_dict[item] = list(accountcode_set).index(item)
print len(list(card_id_set))
#for item in list(card_id_set):
#    card_id_dict[item] = list(card_id_set).index(item)
#%%
'''modify categorial feature to number in data set'''
for item in x:
    item[0] = issuercountry_dict[item[0]]
    item[1] = txvariantcode_dict[item[1]]
  #  item[4] = currencycode_dict[item[4]]
    item[4] = shoppercountry_dict[item[4]]
    item[5] = interaction_dict[item[5]]
    item[6] = verification_dict[item[6]]
    item[9] = accountcode_dict[item[9]]
    
x_array = np.array(x)
y_array = np.array(y)


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
def rf_train(x_train,y_train,x_test,y_test,x_array,y_array):
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
    print 'TP: ' + str(TP)
    print 'FP: ' + str(FP)
    print 'FN: ' + str(FN)
    print 'TN: ' + str(TN)
    print confusion_matrix(y_test, y_predict) #watch out the element in confusion matrix
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
    print('precision: '+str(precision))
    print('recall: ' + str(recall))
    scores = cross_val_score(clf, x_array, y_array, cv=10)
    print '10 fold cv'+ str(scores)
    

    #predict_proba = clf_r.predict_proba(x_test)  # the probability of each smple labelled to positive or negative

#    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#    print('\n')

'''ds1 result'''
print("DS1:")
rf_train(X_train_ds1,Y_train_ds1,x_test,y_test,x_array,y_array)
print("DS2:")
rf_train(X_train_ds2,Y_train_ds2,x_test,y_test,x_array,y_array)
print("DS3:")
rf_train(X_train_ds3,Y_train_ds3,x_test,y_test,x_array,y_array)
print("DS4:")
rf_train(X_train_ds4,Y_train_ds4,x_test,y_test,x_array,y_array)
#%%
def sm_sample(x_train,y_train,x_test,y_test,x_array,y_array):
    sm = SMOTE()
    # x_array = np.array(x)
    # y_array = np.array(y)
    usx = x_train.astype('float64')
    usy = y_train.astype('float64')

    X_res, y_res = sm.fit_sample(usx, usy)
    print(format(Counter(y_train)))
    print('Resampled dataset shape {}'.format(Counter(y_res)))

    print("Random Forest: ")
    rf_train(X_res,y_res,x_test,y_test,x_array,y_array)


    # print("SVM:")
    # svm_train(X_res, y_res, x_test, y_test)

sm_sample(X_train_ds1,Y_train_ds1,x_test,y_test)
sm_sample(X_train_ds2,Y_train_ds2,x_test,y_test)
sm_sample(X_train_ds3,Y_train_ds3,x_test,y_test)
sm_sample(X_train_ds4,Y_train_ds4,x_test,y_test)
#%%
'''
from sklearn.ensemble import IsolationForest
def iForest(X_train,X_test,y_train,y_test):
    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=100, random_state=rng, max_features=13)
    clf.fit(X_train,y_train)
    #y_predct = clf.predict(X_train)
    y_predict = clf.predict(X_test)
    
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
    print 'TP: ' + str(TP)
    print 'FP: ' + str(FP)
    print 'FN: ' + str(FN)
    print 'TN: ' + str(TN)
    print confusion_matrix(y_test, y_predict) #watch out the element in confusion matrix
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
    print('precision: '+str(precision))
    print('recall: ' + str(recall))

iForest(X_train_ds1,Y_train_ds1,x_test,y_test)
'''
#%%import tensorflow as tf
#save tranformed data
'''   
import csv
# fraud_card that has only one time of transition that could not be detect by
#fraud_card_l = fraud_card.tolist()

with open('transformed_data.csv','w') as f:
    f_csv = csv.writer(f)
    #f_csv.writerow(names)
    
    for i in xrange(len(x)-1):
        a=np.append(x_array[i],y[i])
        a=a.tolist()
        f_csv.writerows([a])

'''
















