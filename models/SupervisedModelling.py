#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:24:57 2019

@author: bibhuti
"""

from sklearn.linear_model import LogisticRegression
from models import PeformanceMetrics as PF
from collections import OrderedDict
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


class SupervisedModelling(object):
    
    def __init__(self, X_train,y_train, X_test, y_test):
        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_test = X_test
        self.__y_test = y_test
        self.modelmetrics = {}
        
    def _logistics(self):
        
        logreg = LogisticRegression()
        # fit the model with data
        y_pred = logreg.fit(self.__X_train,self.__y_train).predict(self.__X_test)
        metrics = PF.PerformanceMetrics(self.__y_test, y_pred)
        print(metrics.confMatrix())
        perf_metrics = OrderedDict()
        perf_metrics['ConfusionMatrix'] = metrics.confMatrix()
        perf_metrics['Accuracy'] = "%.2f" % metrics.accuracyScore()
        perf_metrics['Precision'] = "%.2f" % metrics.precisionScore()
        perf_metrics['Recall'] = "%.2f" % metrics.recallScore()
        perf_metrics['AUCScore'] = "%.2f" % metrics.AUCScore(plot=True)
        
        return {'model' : logreg, 'metrics' : perf_metrics}               

    def _naiveBayes(self):    
        gnb = GaussianNB()
        y_pred = gnb.fit(self.__X_train,self.__y_train).predict(self.__X_test)
        metrics = PF.PerformanceMetrics(self.__y_test, y_pred)
        print(metrics.confMatrix())
        perf_metrics = OrderedDict()
        perf_metrics['ConfusionMatrix'] = metrics.confMatrix()
        perf_metrics['Accuracy'] = "%.2f" % metrics.accuracyScore()
        perf_metrics['Precision'] = "%.2f" % metrics.precisionScore()
        perf_metrics['Recall'] = "%.2f" % metrics.recallScore()
        perf_metrics['AUCScore'] = "%.2f" % metrics.AUCScore(plot=True)
        
        return {'model' : gnb, 'metrics' : perf_metrics}               


    def _svmLinear(self):
        clf = SVC(kernel = 'linear',random_state=0, tol=1e-5)
        y_pred = clf.fit(self.__X_train,self.__y_train).predict(self.__X_test)
        metrics = PF.PerformanceMetrics(self.__y_test, y_pred)
        print(metrics.confMatrix())
        perf_metrics = OrderedDict()
        perf_metrics['ConfusionMatrix'] = metrics.confMatrix()
        perf_metrics['Accuracy'] = "%.2f" % metrics.accuracyScore()
        perf_metrics['Precision'] = "%.2f" % metrics.precisionScore()
        perf_metrics['Recall'] = "%.2f" % metrics.recallScore()
        perf_metrics['AUCScore'] = "%.2f" % metrics.AUCScore(plot=True)
        
        return {'model' : clf, 'metrics' : perf_metrics}               
        
    def _svmRadial(self):
        svm = SVC(kernel='rbf', random_state=0, gamma=.01, C=1)
        y_pred = svm.fit(self.__X_train,self.__y_train).predict(self.__X_test)
        metrics = PF.PerformanceMetrics(self.__y_test, y_pred)
        print(metrics.confMatrix())
        perf_metrics = OrderedDict()
        perf_metrics['ConfusionMatrix'] = metrics.confMatrix()
        perf_metrics['Accuracy'] = "%.2f" % metrics.accuracyScore()
        perf_metrics['Precision'] = "%.2f" % metrics.precisionScore()
        perf_metrics['Recall'] = metrics.recallScore()
        perf_metrics['AUCScore'] = metrics.AUCScore(plot=True)
        return {'model' : svm, 'metrics' : perf_metrics}        

    def _decissionTree(self):
        clf = DecisionTreeClassifier(criterion = 'entropy')
        y_pred = clf.fit(self.__X_train,self.__y_train).predict(self.__X_test)
        metrics = PF.PerformanceMetrics(self.__y_test, y_pred)
        print(metrics.confMatrix())
        perf_metrics = OrderedDict()
        perf_metrics['ConfusionMatrix'] = metrics.confMatrix()
        perf_metrics['Accuracy'] = "%.2f" % metrics.accuracyScore()
        perf_metrics['Precision'] = "%.2f" % metrics.precisionScore()
        perf_metrics['Recall'] = metrics.recallScore()
        perf_metrics['AUCScore'] = metrics.AUCScore(plot=True)
        return {'model' : clf, 'metrics' : perf_metrics}        

    def _randomForest(self):
        clf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                     random_state=0)
        y_pred = clf.fit(self.__X_train,self.__y_train).predict(self.__X_test)
        metrics = PF.PerformanceMetrics(self.__y_test, y_pred)
        print(metrics.confMatrix())
        perf_metrics = OrderedDict()
        perf_metrics['ConfusionMatrix'] = metrics.confMatrix()
        perf_metrics['Accuracy'] = "%.2f" % metrics.accuracyScore()
        perf_metrics['Precision'] = "%.2f" % metrics.precisionScore()
        perf_metrics['Recall'] = metrics.recallScore()
        perf_metrics['AUCScore'] = metrics.AUCScore(plot=True)
        return {'model' : clf, 'metrics' : perf_metrics}        
        
    def _AdaBoost(self):
        clf = AdaBoostClassifier(n_estimators=100, random_state=0, 
                                 learning_rate = 1.636)
        y_pred = clf.fit(self.__X_train,self.__y_train).predict(self.__X_test)
        metrics = PF.PerformanceMetrics(self.__y_test, y_pred)
        print(metrics.confMatrix())
        perf_metrics = OrderedDict()
        perf_metrics['ConfusionMatrix'] = metrics.confMatrix()
        perf_metrics['Accuracy'] = "%.2f" % metrics.accuracyScore()
        perf_metrics['Precision'] = "%.2f" % metrics.precisionScore()
        perf_metrics['Recall'] = metrics.recallScore()
        perf_metrics['AUCScore'] = metrics.AUCScore(plot=True)
        return {'model' : clf, 'metrics' : perf_metrics}        
        

    def modelAll(self):
       self.modelmetrics['logistics'] = self._logistics()['metrics']
       self.modelmetrics['svmLinear'] = self._svmLinear()['metrics']        
       self.modelmetrics['svmRadial'] = self._svmRadial()['metrics']
       self.modelmetrics['naivebayes'] = self._naiveBayes()['metrics']
       self.modelmetrics['decissionTree'] = self._decissionTree()['metrics']
       self.modelmetrics['randomforest'] = self._randomForest()['metrics']
       self.modelmetrics['adaboost'] = self._AdaBoost()['metrics']
        
       return self
        
    
    
        
        
        
        
        
        
        
        
    
    
    

