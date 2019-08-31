from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import mean_squared_error

import pandas as pd 
import numpy as np 
import os 
import csv
class  ChurnPredWithGBDT:
	"""docstring for  ChurnPredWithGBDT"""
	def __init__(self):
		
		self.file = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
		self.data = self.feature_transform()
		self.train, self.test = self.split_data()
	
	def isNone(self, values): # 0 填充
		if values == "" or values is None:
			return "0.0"
		else:
			return values

	def feature_transform(self):
		feature_dict = { "gender":{"Male": "1", "Female": "0"}, #建立映射规则
		         "Partner": {"Yes":"1","No":"0"},
		         "Dependents": {"Yes":"1","No":"0"},
		         "PhoneService": {"Yes":"1","No":"0"},
		         "MultipleLines": {"Yes":"1","No":"0","No phone service":"2"},
		         "InternetService":{"DSL":"1","Fiber optic":"2", "No":"0"},
		         "OnlineSecurity":{"Yes":"1","No":"0","No internet service":"2"},
		         "OnlineBackup":{"Yes":"1","No":"0","No internet service":"2"},
		         "DeviceProtection":{"Yes":"1","No":"0","No internet service":"2"},
		         "TechSupport":{"Yes":"1","No":"0","No internet service":"2"},
		         "StreamingTV":{"Yes":"1","No":"0","No internet service":"2"},
		         "StreamingMovies":{"Yes":"1","No":"0","No internet service":"2"},
		         "Contract":{"One year":"1","Month-to-month":"0","Two year":"2"},
		         "PaperlessBilling":{"Yes":"1","No":"0"},
		         "PaymentMethod":{"Electronic check":"0","Mailed check":"1","Bank transfer (automatic)":"2","Credit card (automatic)":"3"},
		         "Churn":{"Yes":"1","No":"0"}
		}

		# df = pd.read_csv(self.file)
		# df.replace(to_replace=r'^\s*$',value=np.nan,regex=True,inplace=True) #空串转成NA
		# df = df.fillna(value="0.0") # 缺失值的处理

		# for column in feature_dict: #字符串映射成数字
		# 	df[column] = df[column].map(feature_dict[column]) 
		
		# df.to_csv("../data/dataMapped.csv")
		# ############################################################
		# # one hot 编码
		# flag = True   
		# for column in df.columns:
		# 	if column not in feature_dict:

		# 		if flag:
		# 			data = pd.DataFrame(df[column])
		# 			flag = False
		# 		else:
		# 			data = data.join(pd.DataFrame(df[column]))
		# 	else:
		# 		if column == "Churn":
		# 			data = data.join(pd.DataFrame(df[column]))
		# 		else:
		# 			newdf = pd.DataFrame(df[column])
		# 			data = data.join(pd.get_dummies(newdf))
		# ######################################################################
		# data.to_csv("../data/dataMapped.csv")
		return pd.read_csv("../data/dataMapped.csv")
	def split_data(self):
		# for column in self.data.columns:
			
		# 	if column != "customerID":
		# 		print(f"{column} finished ! ")
		# 		self.data[column] = self.data[column].astype(float)
		train, test = train_test_split(self.data,test_size = 0.2,random_state=40)
		
		return train,test
	def train_model(self):
		features = [column for column in self.train.columns if column not in ["Churn","customerID"] ] 
		label    = "Churn"
		train_X  = self.train[features]
		train_Y  = self.train[label]

		gbdt = GradientBoostingClassifier()
		gbdt.fit(train_X, train_Y)

		lr = LogisticRegression(penalty='l2',solver='saga',max_iter=1000000)
		lr.fit(train_X, train_Y)
		
		gbdt_lr = LogisticRegression(penalty='l2',solver='saga',max_iter=1000000)
		
		onehot  = OneHotEncoder(categories='auto')
		onehot.fit(gbdt.apply(train_X).reshape(-1,100))
		X_gbdt = onehot.transform(gbdt.apply(train_X).reshape(-1,100))

		gbdt_lr.fit(X_gbdt,train_Y)
	
		return gbdt,lr,(onehot,gbdt_lr)
	def evaluate(self,models):
		#gbdt,lr,gbdt_lr = models
		results = []
		features = [column for column in self.train.columns if column not in ["Churn","customerID"] ]
		label    = "Churn"
		test_X   = self.test[features]
		test_Y   = self.test[label]

		for model in models:
			if model != models[-1]:
				pred_Y   = model.predict_proba(test_X.values)
			
			else:
				pred_Y   = model[1].predict_proba(\
					model[0].transform( models[0].apply(test_X).reshape(-1,100) )
				)
			pred_Y   = list(map(lambda x: 1 if x[1] > 0.5 else 0 ,pred_Y))
			
			mse = mean_squared_error(test_Y.values,pred_Y)
			accuracy = metrics.accuracy_score(test_Y.values,pred_Y)
			auc = metrics.roc_auc_score(test_Y.values,pred_Y)
			results.append((mse,accuracy,auc))

		return results

if __name__ == "__main__":
	pred = ChurnPredWithGBDT()
	pred.feature_transform()
	
	gbdt,lr,gbdt_lr = pred.train_model()
	#print(gbdt.classes_)
	results = pred.evaluate((gbdt,lr,gbdt_lr))
	for result in results:
		print(result) 
	#print(mse, accuracy, auc)