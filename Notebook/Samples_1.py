import class_function as cf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

class Multi_XGB(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        pass
    
    def train(self, x_train, y_train):
        data = pd.concat([x_train, y_train], axis=1)
        group_result = data.groupby(y_train.name)
        data_0 = group_result.get_group(0)
        data_1 = group_result.get_group(1)
        #0比1多
        im_dergee = int(data_0.shape[0]/data_1.shape[0])
        self.im_dergee = im_dergee
        data_list = []
        maj_number = data_0.shape[0]
        balanced_number = data_1.shape[0]
        for i in range(im_dergee):
            if i < im_dergee - 1:
                data_list.append(data_0.iloc[i*balanced_number:(i+1)*balanced_number, :])
            else:
                data_list.append(data_0.iloc[i*balanced_number:, :])
        
        #print(data_list[0])
                
        xgb_cla_list = []
        for i in range(im_dergee):
            x_y = pd.concat([data_list[i], data_1], axis=0)
            #print(x_y)
            x_train_ = x_y.iloc[:, :x_y.shape[1]-1]
            y_train_ = x_y.iloc[:, x_y.shape[1]-1]
            #print(y_train_)
            xgb_cla_list.append(xgb.XGBClassifier(**self.kwargs))
            xgb_cla_list[i].fit(x_train_, y_train_)
            #print(xgb_cla_list[i].classes_)
        
        self.xgb_cla_list = xgb_cla_list
        
    def predict(self, x_test, threshold):
        xgb_cla_list = self.xgb_cla_list
        predict_proba_list = []
        for i in range(self.im_dergee):
            predict_proba = xgb_cla_list[i].predict_proba(x_test)
            predict_proba_1 = []
            for j in range(len(predict_proba)):
                predict_proba_1.append(predict_proba[j][1])
                
            predict_proba_list.append(predict_proba_1)
            
        predict_proba = []
        result = []
        for i in range(len(predict_proba_list[0])):
            tmp = 0
            for j in range(len(predict_proba_list)):
                tmp += predict_proba_list[j][i]
            
            proba = tmp / self.im_dergee
            predict_proba.append(proba)
            if proba > threshold:
                result.append(1)
            else:
                result.append(0)
        
        self.predict_proba = predict_proba
        self.result = result
        
    def cal_K_S(self):
        threshold_list = np.arange(0, 1, 0.01)
        accuracy_list = []
        recall_list = []
        fpr_list = []

        for threshold in threshold_list:
            #y_predicted = multi_xgb.predict(x_test, threshold)
            y_predicted = pd.cut(self.predict_proba, bins = [np.NINF, threshold, 1], labels = [0, 1])
            fbc = cf.For_binary_classifier(y_predicted, y_test)
            accuracy_list.append(fbc.accuracy)
            recall_list.append(fbc.recall)
            fpr_list.append(fbc.fpr)
            
        rec_min_fpr = [recall_list[i] - fpr_list[i] for i in range(len(recall_list))] 
        self.K_S = max(rec_min_fpr)
        self.Acc = accuracy_list[np.argmax(rec_min_fpr)]

data = pd.read_csv('../Data/result_iv_selected.csv')
X = data.drop(columns=['TARGET'])
Y = data.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sam_df = pd.DataFrame()
for i in range(100, x_train.shape[0], 10000):
    multi_xgb = Multi_XGB(max_depth=7, n_estimators=200, random_state=0, colsample_bylevel = 0.2, n_jobs = 8)
    print('training')
    multi_xgb.train(x_train.iloc[: i, :], y_train.iloc[: i])
    print('predicting')
    multi_xgb.predict(x_test, 0.5)
    print('caling')
    multi_xgb.cal_K_S()
    sam_df.loc[i, 'K-S'] = multi_xgb.K_S
    sam_df.loc[i, 'Acc'] = multi_xgb.Acc
    sam_df.to_csv('../Data/sam_1.csv', encoding = 'utf-8_sig')
    print(multi_xgb.K_S, multi_xgb.Acc)
    print('*' * 30)
