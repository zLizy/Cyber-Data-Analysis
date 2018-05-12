from sklearn.datasets import load_iris
import numpy as np  
import pandas as pd  
from sklearn import linear_model  
from sklearn.preprocessing import OneHotEncoder  
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import numpy as np


def read_data(data_file):
	try:
		
		t_X,t_y=load_svmlight_file(data_file)
		return t_X.todense(),t_y
	except ValueError as e:
		print(e)


def oneHot(datasets):
	encode = OneHotEncoder() 
	encode.fit(datasets)
	return encode
	

def gbdt(train_X,train_Y):
	gbdt=GradientBoostingRegressor(n_estimators=500,learning_rate=0.1)
	gbdt.fit(train_X,train_Y)
	return gbdt
	

def gbdt_lr(train_X,train_Y,test_X,test_Y):
	gbdt_model = gbdt(train_X,train_Y)
	tree_feature = gbdt_model.apply(train_X)
	encode = oneHot(tree_feature)
	tree_feature = encode.transform(tree_feature).toarray()

	lr = LogisticRegression()
	lr.fit(tree_feature, train_Y)

	test_X = gbdt_model.apply(test_X)
	tree_feature_test = encode.transform(test_X)
	y_pred = lr.predict_proba(tree_feature_test)[:,1]
# to print fn fp
	y_test=test_Y
	print accuracy_score(y_test, y_pred)
	print confusion_matrix(y_test, y_pred) #watch out the element in confusion matrix
	precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
	print('precision: '+str(precision))
	print('recall: ' + str(recall))

	auc = metrics.roc_auc_score(test_Y, y_pred)
	print "gbdt+lr:",auc
		
def lr(train_X,train_Y,test_X,test_Y):
	lr = LogisticRegression()
	lr.fit(train_X, train_Y)
	y_pred = lr.predict_proba(test_X)[:,1]
	auc = metrics.roc_auc_score(test_Y, y_pred)
	print "only lr:",auc

def gbdt_train(train_X,train_Y,test_X,test_Y):
	model = gbdt(train_X,train_Y)
	y_pred = model.predict(test_X)
	auc = metrics.roc_auc_score(test_Y, y_pred)
	print "only gbdt:",auc

df = pd.read_csv('transformed_data.csv')

x = df.iloc[: ,0:-2].values
y = df.iloc[:, -1].values

print x.shape,y.shape


x_array = np.array(x)
y_array = np.array(y)

X = x_array[:,0:-2]
Y = y_array
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)
gbdt_lr(train_X,train_Y,test_X,test_Y)
lr(train_X,train_Y,test_X,test_Y)
gbdt_train(train_X,train_Y,test_X,test_Y)














