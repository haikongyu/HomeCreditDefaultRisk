import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
import xgboost as xgb
import class_function as cf
import copy
import numpy as np

data = pd.read_csv('../Data/result_iv_selected.csv')
print(data.shape)

X = data.drop(columns = ['TARGET'])
Y = data.TARGET
print(X.shape)

features = list(X.columns)
del_features = []
K_S_list = []
X_0 = copy.deepcopy(X)

while features:
    X_0 = X_0[features]
    K_S_ = []
    i = 0
    for feature in features:
        X_tmp = X_0.drop(columns = [feature])
        
        x_train, x_test, y_train, y_test = train_test_split(X_tmp, Y, test_size=0.2, random_state=0)
        mxg_cla = cf.Multi_XGB(max_depth=7, n_estimators=200, random_state=0, colsample_bylevel = 0.2, n_jobs = 8)
        mxg_cla.train(x_train, y_train)
        mxg_cla.predict(x_test)
        mxg_cla.cal_K_S(y_test)
        K_S_.append(mxg_cla.K_S)
        print(i)
        i += 1
    
    del_feature = features[np.argmax(K_S_)]
    del_features.append(del_feature)
    K_S_list.append(max(K_S_))
    
    df = pd.DataFrame({'del_feature': del_features, 'K-S': K_S_list})
    df.to_csv('../Data/del_features_with_K_S.csv', index = False, encoding = 'utf-8_sig')
    
    features.remove(del_feature)
    print('*' * 10, len(features), '*' * 10)

