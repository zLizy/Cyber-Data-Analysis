import pandas as pd
import numpy as np
import tensorflow as tf
import os
from datetime import datetime 
from sklearn.metrics import roc_auc_score as auc 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = pd.read_csv('transformed_data.csv')
TEST_RATIO = 0.2
df.sort_values('1443650865.0', inplace = True)
TRA_INDEX = int((1-TEST_RATIO) * df.shape[0])
train_x = df.iloc[:TRA_INDEX, 0:-2].values
train_y = df.iloc[:TRA_INDEX, -1].values
print train_x[:,7]
test_x = df.iloc[TRA_INDEX:, 0:-2].values
test_y = df.iloc[TRA_INDEX:, -1].values
cols_mean = []
cols_std = []
for c in range(train_x.shape[1]):
    #
    cols_mean.append(train_x[:,c].mean())
    cols_std.append(train_x[:,c].std())
    train_x[:, c] = (train_x[:, c] - cols_mean[-1]) / cols_std[-1]
    if c==7:
        print train_x[:,c]
    
    test_x[:, c] =  (test_x[:, c] - cols_mean[-1]) / cols_std[-1]

from rbm import RBM
from sklearn.metrics import roc_curve, auc

model = RBM(train_x.shape[1], 10, visible_unit_type='gauss', main_dir='/home/ava/Dropbox/CyberData/ass1/rbm/model', model_name='rbm_model.ckpt',
                 gibbs_sampling_steps=4, learning_rate=0.01, momentum = 0.95, batch_size=500, num_epochs=20, verbose=1)

'''
model.fit(train_x, validation_set=test_x)

test_cost = model.getFreeEnergy(test_x).reshape(-1)
print test_cost, len(test_cost),len(test_y)

print auc(test_y, test_cost)


fpr, tpr, _ = roc_curve(test_y, test_cost)

fpr_micro, tpr_micro, _ = roc_curve(test_y, test_cost)
roc_auc = auc(fpr_micro, tpr_micro)

plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on val data set')
plt.legend(loc="lower right")
#plt.show()
'''
def rbm_train(train_x,test_x,test_y,):
    model.fit(train_x, validation_set=test_x)
    test_cost = model.getFreeEnergy(test_x).reshape(-1)
    print 'the accuracy is'
    #print auc(test_y, test_cost)
    fpr, tpr, _ = roc_curve(test_y, test_cost)

    fpr_micro, tpr_micro, _ = roc_curve(test_y, test_cost)
    roc_auc = auc(fpr_micro, tpr_micro)

    plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve on val data set')
    plt.legend(loc="lower right")
    plt.show()
from imblearn.over_sampling import SMOTE

def sm_sample(x_train,y_train,x_test,y_test):
    sm = SMOTE()
    # x_array = np.array(x)
    # y_array = np.array(y)
    usx = x_train.astype('float64')
    usy = y_train.astype('float64')

    X_res, y_res = sm.fit_sample(usx, usy)
    #print(format(Counter(y_train)))
    #print('Resampled dataset shape {}'.format(Counter(y_res)))

    #rf_train(X_res,y_res,x_test,y_test)
    rbm_train(X_res,x_test,y_test)
sm_sample(train_x,train_y,test_x,test_y)


