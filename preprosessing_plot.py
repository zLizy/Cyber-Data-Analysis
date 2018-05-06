import datetime
import time
import random

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn import neighbors,metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier

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
def aggregate(before_aggregate, aggregate_feature):
    if aggregate_feature == 'day':
        after_aggregate = []
        pos_date = -1
        before_aggregate.sort(key = itemgetter(9))#sort by timestamp
        # print(itemgetter(9))
        #print(itemgetter(-2))
        temp = groupby(before_aggregate, itemgetter(-1))
        group_unit = []
        mean = []
        for i, item in temp:# i is group id
            for jtem in item:# unit in each group
                group_unit.append(jtem)

            # for feature_i in xrange(6):
            #    print(feature_i)
            #    mean.append(zip(group_unit)[feature_i])
            after_aggregate.append(group_unit)
            #after_aggregate.append(mean)
            group_unit = []
        #print after_aggregate[0]
        #print before_aggregate[0]
    if aggregate_feature == 'client':
        after_aggregate = []
        pos_client = -3
        before_aggregate.sort(key = itemgetter(pos_client))#sort with cardID firstly，if sort with 2 feature, itemgetter(num1,num2)
        temp = groupby(before_aggregate, itemgetter(pos_client))#group
        group_unit = []
        for i, item in temp:# i is group id
            for jtem in item:# unit in each group
                group_unit.append(jtem)
            after_aggregate.append(group_unit)
            group_unit = []
    return after_aggregate

def aggregate_mean(before_aggregate):
    #print before_aggregate[0]
    if True:
        after_aggregate = []
        pos_date = -1
        before_aggregate.sort(key = itemgetter(-1))#sort by timestamp
        temp = groupby(before_aggregate, itemgetter(-1))
        group_unit = []
        mean = []
        for i, item in temp:# i is group id
            for jtem in item:# unit in each group
                group_unit.append(list(jtem))
            #print group_unit
            if len(zip(group_unit)) < 2:
                after_aggregate.append(group_unit)
                group_unit = []
            if len(zip(group_unit)) >= 2:
                #print zip(group_unit)
                for feature_i in xrange(14):
                    #print zip(group_unit)[feature_i]
                    mean.append(sum(zip(*group_unit)[feature_i])/len(zip(group_unit)))
                after_aggregate.append(mean)
                group_unit = []
                mean = []
        #print after_aggregate[0]
        #print before_aggregate[0]
    return after_aggregate

src = '/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035_Cyber_Data_Analytics/data/data_for_student_case.csv'
ah = open(src, 'r')

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
    amount = float(line_ah.strip().split(',')[5])#transaction amount in minor units
    currencycode = line_ah.strip().split(',')[6]
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
    # if cvcresponse > 2:
    #     cvcresponse = 3
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

data = sorted(data, key = lambda k: k[-1])

data_all = DataFrame(data)
data_all.columns = ['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode',
                'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp',
                 'accountcode','mail_id', 'ip_id', 'card_id', 'label', 'creationdate']

day_aggregate = aggregate(data,'day')
client_aggregate = aggregate(data,'client')
transaction_num_day = []
Date = []
for item in day_aggregate:
    transaction_num_day.append(len(item))
    date = datetime.datetime.strptime(item[0][-1], '%Y-%m-%d')
    Date.append(date)
plt.figure(1)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(Date,transaction_num_day, color = 'c', linewidth = 2)
plt.xticks(pd.date_range(Date[0], Date[-1], freq='M'))  # 时间间隔
plt.gcf().autofmt_xdate()

plt.plot()
plt.text(2,0.0,'Date: 2015-10-8')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.xlim([0,125])
plt.axis('tight')
plt.savefig('/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035 Cyber Data Analytics (201718 Q4) - 4282018 - 441 PM/Day Aggregating.png')
transaction_num_client = []
for item in client_aggregate:
    transaction_num_client.append(len(item))
    plt.figure(2)
    plt.plot(transaction_num_client, color = 'c', linewidth = 2)
    #plt.text(99,9668,'Date: 2015-10-8')
    plt.xlabel('Client ID')
    plt.ylabel('Number of Transactions')
    plt.axis('tight')
    plt.savefig('Client Aggregating.png')



# retrieve the fraud transactions
fraud = data_all.loc[data_all['label']==1]
fraud_list = fraud.values.tolist()
# compute the frequency of fraud transactions based on date or user
fraud_day_aggregate = aggregate(fraud_list,'day')
fraud_client_aggregate = aggregate(fraud.values.tolist(),'client')

#plot frequency graph based on Date
fraud_transaction_num_day = []
Date = []
for item in fraud_day_aggregate:
    fraud_transaction_num_day.append(len(item))
    date = datetime.datetime.strptime(item[0][-1],'%Y-%m-%d')
    Date.append(date)
    #print(date)
plt.figure(1)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.plot(Date,fraud_transaction_num_day, color = 'c', linewidth = 2)
# plt.xticks(pd.date_range(Date[0], Date[-1], freq='M'))  # 时间间隔
# plt.gcf().autofmt_xdate()
# plt.plot()

frequency = np.array(fraud_transaction_num_day)
fraud_frequency = frequency/float(np.sum(frequency))
plt.hist(fraud_frequency,int(np.sqrt(len(Date))))
plt.savefig('/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035 Cyber Data Analytics (201718 Q4) - 4282018 - 441 PM/Fraud Hist Day Aggregating.png')

plt.text(2,0.0,'Date: 2015-10-8')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.xlim([0,125])
plt.axis('tight')
plt.savefig('/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035 Cyber Data Analytics (201718 Q4) - 4282018 - 441 PM/Fraud Day Aggregating.png')

#plot frequency graph based on user
fraud_transaction_num_client = []
for item in fraud_client_aggregate:
    fraud_transaction_num_client.append(len(item))
    plt.figure(2)
    plt.plot(fraud_transaction_num_client, color = 'c', linewidth = 2)
    #plt.text(99,9668,'Date: 2015-10-8')
    plt.xlabel('Client ID')
    plt.ylabel('Number of Transactions')
    plt.axis('tight')
    plt.savefig('/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035 Cyber Data Analytics (201718 Q4) - 4282018 - 441 PM/Fraud Client Aggregating.png')


# In[6]:
x = []#contains features
y = []#contains labels
for item in data:#split data into x,y
    x.append(item[0:-2])
    y.append(item[-2])

X = x[:]
'''map number to each categorial feature'''
for item in list(issuercountry_set):
    issuercountry_dict[item] = list(issuercountry_set).index(item)
for item in list(txvariantcode_set):
    txvariantcode_dict[item] = list(txvariantcode_set).index(item)
for item in list(currencycode_set):
    currencycode_dict[item] = list(currencycode_set).index(item)
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
'''modify categorial feature to number in data set'''
for item in x:
    item[0] = issuercountry_dict[item[0]]
    item[1] = txvariantcode_dict[item[1]]
    item[4] = currencycode_dict[item[4]]
    item[5] = shoppercountry_dict[item[5]]
    item[6] = interaction_dict[item[6]]
    item[7] = verification_dict[item[7]]
    item[10] = accountcode_dict[item[10]]

#x_mean = []
#x_mean = aggregate_mean(x);

# x_mean = x[:];
# des = '/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035_Cyber_Data_Analytics/data/original_data.csv'
# des1 = '/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035_Cyber_Data_Analytics/data/aggregate_data.csv'
# ch_dfa = open(des,'w')

#ch_dfa.write('txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,'+
#             'currencycode,shoppercountrycode,shopperinteraction,simple_journal,'+
 #            'cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id')
#ch_dfa.write('\n')

# sentence = []
# for i in range(len(x_mean)):
#     for j in range(len(x_mean[i])):
#         sentence.append(str(x_mean[i][j]))
#     sentence.append(str(y[i]))
#     ch_dfa.write(' '.join(sentence))
#     ch_dfa.write('\n')
#     sentence=[]
#     ch_dfa.flush()
#


TP, FP, FN, TN = 0, 0, 0, 0
x_array = np.array(x)
y_array = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size = 0.2)#test_size: proportion of train/test data

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


def plot_roc(predict_proba,y_test,y_predict,idx):
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
    print confusion_matrix(y_test, y_predict)  # watch out the element in confusion matrix
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))

    # y_test = y_test.reshape((47340,1))
    auc = metrics.roc_auc_score(y_test, predict_proba)
    fpr, tpr, _ = metrics.roc_curve(y_test, predict_proba)
    #plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label="data" + idx + ", auc=" + str(auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig("/Users/lizy/my_doc/Q4/cyber_data_analysis/Cyber-Data-Analysis/plot/DS-"+idx+".png")


'''SVM'''
def svm_train(x_train,y_train,x_test,y_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    # clf = neighbors.KNeighborsClassifier(algorithm = 'kd_tree')
    # clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print("svm accuracy: ")
    print accuracy_score(y_test, y_predict)
    idx = 'none'
    predict_proba = clf.predict_proba(x_test)  # the probability of each smple labelled to positive or negative
    plot_roc(predict_proba,y_test,y_predict,idx)

    print('\n')

'''ds1 result'''
svm_train(X_train_ds1,Y_train_ds1,x_test,y_test)
svm_train(X_train_ds2,Y_train_ds2,x_test,y_test)
svm_train(X_train_ds3,Y_train_ds3,x_test,y_test)
svm_train(X_train_ds4,Y_train_ds4,x_test,y_test)

'''Random Forest'''
def rf_train(x_train,y_train,x_test,y_test,idx):
    clf_r = RandomForestClassifier(max_depth=2, random_state=0)
    clf_r.fit(x_train, y_train)
    y_predict = clf_r.predict(x_test)
    print("random forest accuracy: ")
    print accuracy_score(y_test, y_predict)
    predict_proba = clf_r.predict_proba(x_test)[:, 1]  # the probability of each smple labelled to positive or negative
    plot_roc(predict_proba,y_test,y_predict,idx)
    print('\n')

'''ds1 result'''
print("DS1:")
rf_train(X_train_ds1,Y_train_ds1,x_test,y_test,'1')
print("DS2:")
rf_train(X_train_ds2,Y_train_ds2,x_test,y_test,'2')
print("DS3:")
rf_train(X_train_ds3,Y_train_ds3,x_test,y_test,'3')
print("DS4:")
rf_train(X_train_ds4,Y_train_ds4,x_test,y_test,'4')


'''Neural Network '''
def nn_train(x_train,y_train,x_test,y_test,idx):
    x_train = x_train.astype('float64')
    y_train = y_train.astype('float64')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5,5), random_state=1)
    clf.fit(x_train, y_train)
    x_test = x_test.astype('float64')
    y_test = y_test.astype('float64')
    y_predict = clf.predict(x_test)
    print("random forest accuracy: ")
    print accuracy_score(y_test, y_predict)
    predict_proba = clf.predict_proba(x_test)[:, 1]  # the probability of each smple labelled to positive or negative
    plot_roc(predict_proba, y_test, y_predict, idx)
    print('\n')

print("DS1:")
nn_train(X_train_ds1,Y_train_ds1,x_test,y_test,'1(NN)')
print("DS2:")
nn_train(X_train_ds2,Y_train_ds2,x_test,y_test,'2(NN)')
print("DS3:")
nn_train(X_train_ds3,Y_train_ds3,x_test,y_test,'3(NN)')
print("DS4:")
nn_train(X_train_ds4,Y_train_ds4,x_test,y_test,'4(NN)')


#data_all = pd.read_csv('/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035 Cyber Data Analytics (201718 Q4) - 4282018 - 441 PM/data_for_student_case.csv')

#data_all.to_csv('/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035_Cyber_Data_Analytics/data/data.csv')



'''=========== RESAMPLE SMOTE =============='''
def sm_sample(x_train,y_train,x_test,y_test,idx):
    sm = SMOTE()
    # x_array = np.array(x)
    # y_array = np.array(y)
    usx = x_train.astype('float64')
    usy = y_train.astype('float64')

    X_res, y_res = sm.fit_sample(usx, usy)
    print(format(Counter(y_train)))
    print('Resampled dataset shape {}'.format(Counter(y_res)))

    print("Random Forest: ")
    rf_train(X_res,y_res,x_test,y_test,idx)
    #nn_train(X_res,y_res,x_test,y_test,idx)
    # print("SVM:")
    # svm_train(X_res, y_res, x_test, y_test)
plt.figure()
sm_sample(X_train_ds1,Y_train_ds1,x_test,y_test,'SM-1')
sm_sample(X_train_ds2,Y_train_ds2,x_test,y_test,'SM-2')
sm_sample(X_train_ds3,Y_train_ds3,x_test,y_test,'SM-3')
sm_sample(X_train_ds4,Y_train_ds4,x_test,y_test,'SM-4')

plt.figure()
rf_train(X_train_ds1,Y_train_ds1,x_test,y_test,'1(ORI)')
sm_sample(X_train_ds1,Y_train_ds1,x_test,y_test,'1(SMOTE)')

plt.figure()
rf_train(X_train_ds2,Y_train_ds2,x_test,y_test,'2(ORI)')
sm_sample(X_train_ds2,Y_train_ds2,x_test,y_test,'2(SMOTE)')

plt.figure()
rf_train(X_train_ds3,Y_train_ds3,x_test,y_test,'3(ORI)')
sm_sample(X_train_ds3,Y_train_ds3,x_test,y_test,'3(SMOTE)')

plt.figure()
rf_train(X_train_ds3,Y_train_ds3,x_test,y_test,'3(ORI)')
sm_sample(X_train_ds3,Y_train_ds3,x_test,y_test,'3(SMOTE)')

plt.figure()
nn_train(X_train_ds1,Y_train_ds1,x_test,y_test,'1(nn-ORI)')
sm_sample(X_train_ds1,Y_train_ds1,x_test,y_test,'1(nn-SMOTE)')

# clf = neighbors.KNeighborsClassifier(algorithm = 'kd_tree')
