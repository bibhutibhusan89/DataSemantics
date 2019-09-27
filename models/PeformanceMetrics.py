#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:32:45 2019

@author: bibhuti
"""

from sklearn import metrics
import matplotlib.pyplot as plt


class PerformanceMetrics(object):
    
    def __init__(self, y_test, y_pred):
        
        self.__y_test = y_test
        self.__y_pred = y_pred
        self.__confMat = metrics.confusion_matrix(self.__y_test, self.__y_pred)
        self.__accuracy = metrics.accuracy_score(self.__y_test, self.__y_pred)
        self.__precision = metrics.precision_score(self.__y_test, self.__y_pred)
        self.__recall = metrics.recall_score(self.__y_test, self.__y_pred)
        self.__fpr, self.__tpr, _ = metrics.roc_curve(self.__y_test,  self.__y_pred)
        self.__auc = round(metrics.roc_auc_score(self.__y_test, self.__y_pred),3)
       
        
    def confMatrix(self):
        return self.__confMat
    
    def accuracyScore(self):
        return self.__accuracy
    
    def precisionScore(self):
        return self.__precision
        
    def recallScore(self):
        return self.__recall
        
        
    def AUCScore(self, plot = True):
        if(plot):
            plt.plot(self.__fpr,self.__tpr,label="AUC Area="+str(self.__auc))
            plt.legend(loc=4)
            plt.show()
    
        return self.__auc        
        
        