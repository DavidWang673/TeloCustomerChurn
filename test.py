import pandas as pd 
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
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = df.fillna(value="0.0") # 缺失值的处理
########################################################
for column in feature_dict: #字符串映射成数字
	df[column] = df[column].map(feature_dict[column]) 
#############################################################
## one hot 编码
# flag = True   
# for column in df.columns:
# 	if column not in feature_dict:
# 		if flag:
# 			data = pd.DataFrame(df[column])
# 			flag = False
# 		else:
# 			data = data.join(pd.DataFrame(df[column]))
# 	else:
# 		newdf = pd.DataFrame(df[column])
# 		data = data.join(pd.get_dummies(newdf))
#######################################################################