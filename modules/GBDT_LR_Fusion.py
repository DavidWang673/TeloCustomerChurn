from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
#from sklearn.metrics import mean_squared_error

import pandas as pd 
import numpy as np 

data = pd.read_csv("../data/dataMapped.csv")

features = [f for f in data.columns if f not in ["Unnamed: 0","customerID","Churn"] ]
label = [l for l in data.columns if l == "Churn"]

X = data[features]
Y = data[label]

n_samples   = len(Y)
n_positive  = len(Y[Y["Churn"] == 1])
n_negative  = n_samples - n_positive
if abs(n_positive/n_negative - 1) <0.2:
	print('balanced data')
else:
	print("imbalanced data, do not choose 'accuracy' as metric")

n_estimators = 100
gbdt = GradientBoostingClassifier(n_estimators = n_estimators)
lr   = LogisticRegression(solver='saga',n_jobs=-1,max_iter=10000) 


X_train, X_test, Y_train, Y_test = train_test_split(\
	X, Y, test_size=0.3, random_state=0)

X_train, X_test, Y_train, Y_test = X_train.values, X_test.values, Y_train.values.reshape(-1), Y_test.values.reshape(-1)


gbdt.fit(X_train,Y_train)

onehot = OneHotEncoder(categories='auto')
onehot.fit(gbdt.apply(X_train).reshape(-1,n_estimators))
X_train_new = onehot.transform(gbdt.apply(X_train).reshape(-1,n_estimators))

lr.fit(X_train_new,Y_train)



########### Evaluations #####################
Y_pred = lr.predict_proba(
						   onehot.transform(gbdt.apply(X_test).reshape(-1,n_estimators))
	                     ) 
Y_pred   = list(map(lambda x: 1 if x[1] > 0.5 else 0 ,Y_pred))

accuracy = metrics.balanced_accuracy_score(Y_pred,Y_test)
precision = metrics.precision_score(Y_pred,Y_test)
recall = metrics.recall_score(Y_pred,Y_test)
f1 = metrics.f1_score(Y_pred,Y_test)
auc = metrics.roc_auc_score(Y_pred,Y_test)
#print(auc)
print(f"acc: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc}")
