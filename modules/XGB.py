
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn import metrics
import matplotlib.pyplot as plt, graphviz 

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

###################################
X_train, X_test, Y_train, Y_test = train_test_split(\
	X, Y, test_size=0.3, random_state=0)

X_train, X_test, Y_train, Y_test = X_train.values, X_test.values, Y_train.values.reshape(-1), Y_test.values.reshape(-1)
########################################################
# pca = PCA(n_components=10)
# pca.fit(X)
# X_train = pca.transform(X_train)
# X_test  = pca.transform(X_test)
#######################################################
train = xgb.DMatrix( X_train, Y_train)
test  = xgb.DMatrix( X_test,  Y_test)
num_round = 18
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = ['auc']
evallist = [(test, 'eval'), (train, 'train')]
bst = xgb.train( param, train, num_round, evallist )
Y_pred = bst.predict(test)
Y_pred = list(map(lambda x: 1 if x>0.5 else 0,Y_pred))

accuracy = metrics.balanced_accuracy_score(Y_pred,Y_test)
precision = metrics.precision_score(Y_pred,Y_test)
recall = metrics.recall_score(Y_pred,Y_test)
f1 = metrics.f1_score(Y_pred,Y_test)
auc = metrics.roc_auc_score(Y_pred,Y_test)
#print(auc)
print(f"acc: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc}")

xgb.plot_importance(bst)
plt.show()
xgb.plot_tree(bst, num_trees=2)
plt.show()
