from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
#from sklearn.metrics import mean_squared_error

import pandas as pd 
import numpy as np 
import os 


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

gbdt = GradientBoostingClassifier()

X_train, X_test, Y_train, Y_test = train_test_split(\
	X, Y, test_size=0.3, random_state=0)
X_train, X_test, Y_train, Y_test = X_train.values, X_test.values, Y_train.values.reshape(-1), Y_test.values.reshape(-1)


# CV 选择最佳参数
tuned_parameters = {'learning_rate': [0.001,0.01,0.1],
                     'n_estimators': [20, 25, 30, 35,40,50,60,70,80],
                     'min_samples_split':[2,5,8,10],
                     'max_depth':[2,3,4,5,6,7],
                     'n_iter_no_change' : [None,20] #early_stop
                     #'C': [1, 10, 100, 1000]
                     }

scores = ['balanced_accuracy','precision', 'recall','roc_auc' ]
#'balanced_accuracy' implemented in sklearn to deal with the imbalanced data 
for score in scores:
    print(f"# Tuning hyper-parameters for {score}" )
    print()

    clf = GridSearchCV(gbdt, tuned_parameters, cv=10,
                       scoring=score )
    clf.fit(X_train, Y_train)

    print("Best parameters set found on trainning set:")
    print(clf.best_params_)
    print()
    print("Grid scores on trainning set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print(f"{mean:.3} (+/-{(std * 2):.3}) for {params}")
    print()
    print()
    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
