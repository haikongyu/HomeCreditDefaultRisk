import pandas as pd
import numpy as np


class For_binary_classifier(object):
    def __init__(self, y_predicted, y_real):
        df = pd.DataFrame({'y_predicted':y_predicted, 'y':y_real})
        df = df.reset_index().drop(columns='index')
        TP = 0
        FP = 0
        TN = 0
        FN = 0
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
        '''
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
