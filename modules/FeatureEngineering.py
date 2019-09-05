import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,LabelBinarizer,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

df = pd.read_csv("E:/Res/TelcoCustomerChurn/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 缺失值的处理
df.replace(to_replace=r'^\s*$',value=np.nan,regex=True,inplace=True) #空串转成NA
df = df.fillna(value="0.0") 

label = "Churn"
ID    = "customerID"

# 特征分桶 "tenure","MonthlyCharges","TotalCharges" (连续型特征的直方图)
features_bucketed = ["tenure","MonthlyCharges","TotalCharges"]
groups            = [0,0,0]

sc = StandardScaler() #去量纲
for g,fea in zip(groups, features_bucketed):
	#################  画直方图 #########################
	# df[fea].astype("float").plot.hist(grid=True, bins=20, rwidth=0.9,
    #                      color='#607c8e')
	# plt.show()
	#########################################
	#f[fea+"bucketed"] = pd.qcut(df[fea], 20)  # 等频分箱
	df[fea+"bucketed"] = pd.cut(df[fea].astype("float"), 5)   # 等距分箱，频数不一定相等
	#df[fea]            = sc.fit_transform(df[fea].astype("float"))
df[features_bucketed] = sc.fit_transform(df[features_bucketed].astype('float'))
order = [ f for f in df.columns if f != label ]
order.append(label)
df = df[order]



################################################################################################
features_str = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',
                    'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                    'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
                    'tenurebucketed', 'MonthlyChargesbucketed', 'TotalChargesbucketed','Churn'
                   ]
le = LabelEncoder()  #字符串类型全部啊转化为数字类型
for fea in features_str:
	df[fea] = le.fit_transform(df[fea])
################################################################################################

######################### OneHotEncoder ########################################################################
features_dummies = [f for f in df.columns if f not in [ID,label,"tenure","MonthlyCharges","TotalCharges"]]

df_onehot = pd.get_dummies(data=df,columns=features_dummies)

#df_onehot.to_csv("E:/Res/TelcoCustomerChurn/data/FE.csv")
#########################################################################################
features = [f for f in df_onehot.columns if f not in ["Unnamed: 0","customerID","Churn"] ]
label = [l for l in df_onehot.columns if l == "Churn"]

X = df_onehot[features]
Y = df_onehot[label]

n_samples   = len(Y)
n_positive  = len(Y[Y["Churn"] == 1])
n_negative  = n_samples - n_positive
if abs(n_positive/n_negative - 1) <0.2:
	print('balanced data')
else:
	print("imbalanced data, do not choose 'accuracy' as metric")

lr = LogisticRegression(n_jobs=-1,max_iter=10000,solver='saga')

X_train, X_test, Y_train, Y_test = train_test_split(\
	X, Y, test_size=0.3, random_state=0)
X_train, X_test, Y_train, Y_test = X_train.values, X_test.values, Y_train.values.reshape(-1), Y_test.values.reshape(-1)


lr.fit(X_train, Y_train)

Y_pred = lr.predict_proba(
						   X_test
	                     ) 
Y_pred   = list(map(lambda x: 1 if x[1] > 0.5 else 0 ,Y_pred))

accuracy = metrics.balanced_accuracy_score(Y_pred,Y_test)
precision = metrics.precision_score(Y_pred,Y_test)
recall = metrics.recall_score(Y_pred,Y_test)
f1 = metrics.f1_score(Y_pred,Y_test)
auc = metrics.roc_auc_score(Y_pred,Y_test)
#print(auc)
print(f"acc: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc}")

# df_onehot = pd.DataFrame(df[ID])
# for f in df.columns:
# 	if f not in features_dummies: 
# 		if f != ID:
# 			df_onehot = df_onehot.join(pd.DataFrame(df[f]))
# 	else:
# 		df_onehot.join(pd.get_dummies(pd.DataFrame(df[f])))
# print(df_onehot.columns)
#features_dummies.extend
# sub_fea = [f for f in df.columns if f not in features_dummies]
# print( sub_fea)
# df_onehot = pd.get_dummies(df[features_dummies])
# print(df_onehot.columns)
#df_onehot.to_csv("E:/Res/TelcoCustomerChurn/data/FE.csv")
# print(AAAAAAA)
# df_after  = df_onehot.join(df[[f for f in df.columns if f not in features_dummies and f not in [label,ID]]])
# df_after  = pd.DataFrame(df[ID]).join(df_after).join(pd.DataFrame(df[label]))
# df_after.to_csv("E:/Res/TelcoCustomerChurn/data/FE.csv")
# onehot = OneHotEncoder(categories='auto')
# print(len(df[features_dummies].columns))
# df_onehot = onehot.fit_transform(df[features_dummies])
# df_onehot = pd.DataFrame(df_onehot,)
# print(type(df_onehot))
####################################################################################
# lb = LabelBinarizer()  #没办法处理区间(特征分桶添加的特征)， 用于对类别二值化
# for fea in features_dummies:
# 	df[fea]= lb.fit_transform(df[fea])
# 	print(df[fea][0])
######################################################################################
