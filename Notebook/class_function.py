import pandas as pd
import numpy as np
import xgboost as xgb

class For_binary_classifier(object):
    def __init__(self, y_predicted, y_real):
        df = pd.DataFrame({'y_predicted':y_predicted, 'y':y_real})
        df = df.reset_index().drop(columns='index')
        TP = df[(df.iloc[:, 0] == 1) & (df.iloc[:, 1] == 1)].shape[0]
        FP = df[(df.iloc[:, 0] == 1) & (df.iloc[:, 1] != 1)].shape[0]
        TN = df[(df.iloc[:, 0] != 1) & (df.iloc[:, 1] != 1)].shape[0]
        FN = df[(df.iloc[:, 0] != 1) & (df.iloc[:, 1] == 1)].shape[0]
        '''
        for i in range(df.shape[0]):
            if df.iloc[i,0]:
                if df.iloc[i, 1]:
                    TP += 1
                else:
                    FP += 1
            else:
                if df.iloc[i,1]:
                    FN += 1
                else:
                    TN += 1
        
        print('{:^15}'.format(''), '{:^15}'.format('Real +'), '{:^15}'.format('Real -'))
        print('{:^15}'.format('Predict +'), '{:^15}'.format(TP), '{:^15}'.format(FP))
        print('{:^15}'.format('Predict -'), '{:^15}'.format(FN), '{:^15}'.format(TN))
        print('')
        '''
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        
        if TP+FN:
            recall = TP/(TP+FN)
        else:
            recall = 0    
        
        if (TP+FP) > 0:
            precision = TP/(TP+FP)
        else:
            precision=0
        
        MissingAlarm = 1 - recall
        
        FalseAlarm = 1 - precision
        
        if precision*recall == 0:
            F1 = 0
        else:
            F1 = (2*precision*recall)/(precision+recall)

        
        if FP+TN:
            fpr = FP/(FP+TN)
        else: 
            fpr = 0

        
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        self.MissingAlarm = MissingAlarm
        self.FalseAlarm = FalseAlarm
        self.F1 = F1
        self.fpr = fpr

class chimerge(object):
    def __init__(self, X, Y, threshold, min_bins):
        self.X = X
        self.Y = Y
        self.Y_unique = Y.unique()
        self.threshold = threshold
        self.min_bins = min_bins
        self.bins = [self.X.min()+i*((self.X.max()-self.X.min()))/20 for i in range(21)]
        self.bins[0] = self.bins[0]-0.01
        self.bins = np.array(self.bins)
        print(type(self.bins))
        self.x = pd.cut(self.X, self.bins)
        
    def cal_rate(self):
        self.combine = pd.concat([self.x, self.Y], axis=1)
        result = self.combine.groupby(self.x.name)
        rate_list = []
        j = 0
        for tup in result:
            rate_list.append([self.bins[j]])
            for col in self.Y_unique:
                if not tup[1].shape[0]:
                    rate = 0
                else:
                    rate = tup[1][tup[1][self.Y.name]==col].shape[0]/tup[1].shape[0]
                rate_list[j].append(rate)
            j += 1
        self.rate_list = rate_list
    def cal_chi(self):
        chi_list = []
        for i in range(len(self.rate_list) - 1):
            chi = 0
            for j in range(len(self.rate_list[0]) - 1):
                chi += (self.rate_list[i][j+1] - self.rate_list[i+1][j+1]) ** 2
            chi_list.append([self.rate_list[i+1][0], chi])
        self.chi_list = chi_list
    def find_delete(self):
        tmp = []
        for i in range(len(self.chi_list)):
            tmp.append(self.chi_list[i][1])
        index = tmp.index(min(tmp))
        self.min_chi = min(tmp)
        self.delete = self.chi_list[index][0]
        index = np.argwhere(self.bins == self.delete)
        self.bins = np.delete(self.bins, index)
    def main(self):
        self.cal_rate()
        self.cal_chi()
        self.find_delete()
        self.x = pd.cut(self.X, self.bins)
        bins_num = len(self.x.unique())
        while bins_num > self.min_bins and self.min_chi < self.threshold:
            print(bins_num, self.min_chi)
            self.cal_rate()
            self.cal_chi()
            self.find_delete()
            self.x = pd.cut(self.X, self.bins)
            bins_num = len(self.x.unique())
            
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
        
    def predict(self, x_test, threshold = 0.5):
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

    def cal_K_S(self, y_test):
        threshold_list = np.arange(0, 1, 0.01)
        accuracy_list = []
        recall_list = []
        fpr_list = []

        for threshold in threshold_list:
            #y_predicted = multi_xgb.predict(x_test, threshold)
            y_predicted = pd.cut(self.predict_proba, bins = [np.NINF, threshold, 1], labels = [0, 1])
            fbc = For_binary_classifier(y_predicted, y_test)
            accuracy_list.append(fbc.accuracy)
            recall_list.append(fbc.recall)
            fpr_list.append(fbc.fpr)
            
        rec_min_fpr = [recall_list[i] - fpr_list[i] for i in range(len(recall_list))] 
        self.K_S = max(rec_min_fpr)
        self.Acc = accuracy_list[np.argmax(rec_min_fpr)]
        
class Chimerge(object):
    #X 待分箱特征，Y,目标变量，init_bins初始等距分箱个数，threshold阈值，mib_bins 最小分箱个数
    def __init__(self, X, Y, init_bins, threshold, min_bins):
        self.X = X
        self.Y = Y
        self.Y_unique = Y.unique()
        self.threshold = threshold
        self.min_bins = min_bins
        self.bins = [self.X.min()+i*((self.X.max()-self.X.min()))/init_bins for i in range(init_bins+1)]
        self.bins[0] = np.NINF
        self.bins = np.array(self.bins)
        #print(type(self.bins))
        self.x = pd.cut(self.X, self.bins)
        
    def cal_chi2(self):
        chi2_list = []
        df = pd.concat([self.X, self.Y], axis=1)
        for i in range(len(self.bins) - 2):
            chi2 = 0
            df_tmp = df[(df.iloc[:, 0] > self.bins[i]) & (df.iloc[:, 0] <= self.bins[i+2])]
            A00 = df_tmp[(df_tmp.iloc[:, 0] <= self.bins[i+1]) & (df_tmp.iloc[:, 1] == 0)].shape[0]
            A01 = df_tmp[(df_tmp.iloc[:, 0] <= self.bins[i+1]) & (df_tmp.iloc[:, 1] == 1)].shape[0]
            A10 = df_tmp[(df_tmp.iloc[:, 0] > self.bins[i+1]) & (df_tmp.iloc[:, 1] == 0)].shape[0]
            A11 = df_tmp[(df_tmp.iloc[:, 0] > self.bins[i+1]) & (df_tmp.iloc[:, 1] == 1)].shape[0]
            R0 = df_tmp[(df_tmp.iloc[:, 0] <= self.bins[i+1])].shape[0]
            R1 = df_tmp[(df_tmp.iloc[:, 0] > self.bins[i+1])].shape[0]
            C0 = df_tmp[df_tmp.iloc[:, 1] == 0].shape[0]
            C1 = df_tmp[df_tmp.iloc[:, 1] == 1].shape[0]
            N = df_tmp.shape[0]
            if N:
                E00 = R0 * C0 / N
                E01 = R0 * C1 / N
                E10 = R1 * C0 / N
                E11 = R1 * C1 / N
                if E00 and E01 and E10 and E11:
                    chi2 = (A00 - E00) ** 2 / E00 + (A01 - E01) ** 2 / E01 + (A10 - E10) ** 2 / E10 + (A11 - E11) ** 2 / E11
                else:
                    chi2 = np.inf
            else:
                chi2 = 0
            chi2_list.append(chi2)
            
        self.chi2_list = chi2_list
        
    def find_delete(self):
        self.min_chi2 = min(self.chi2_list)
        index = np.argmin(self.chi2_list)
        self.bins = np.delete(self.bins, index + 1)
        
    def main(self):
        self.cal_chi2()
        self.find_delete()
        self.x = pd.cut(self.X, self.bins)
        bins_num = len(self.x.unique())
        while bins_num > self.min_bins and self.min_chi2 < self.threshold:
            #print(bins_num, self.min_chi)
            self.cal_chi2()
            self.find_delete()
            self.x = pd.cut(self.X, self.bins)
            bins_num = len(self.x.unique())
        #输出最后分箱个数
        print(bins_num)
