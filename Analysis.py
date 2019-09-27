#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:39:55 2019

@author: bibhuti

Script to Build Model and Deploy in production

"""

import pandas as pd
import numpy as np
from Reader import FileReader as FR
from metrics import DataTransformation as DTS
from models import SupervisedModelling as SM
from sklearn.externals import joblib

path = './data/'
file = path+'training_1.csv'

traindata = FR.FileReader(file, filetype = 'csv').data
print(traindata.head())

""" Replacing Categorical Variables in text column into dummy variable using One hot encoding """

modeldata = pd.DataFrame(DTS.DataTransformation().elementBucket(traindata['text']))


''' split train and test 75 * 25 '''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   modeldata, traindata['category'], test_size=0.25, random_state=42, shuffle =True)


supervised = SM.SupervisedModelling(X_train,y_train, X_test, y_test)
#supervised.modelAll()


''' With multiple iterations of Analysis in Analysis Test file AdaBoost with 
concordance threshold of 0.55 was giving best result of Accuracy 0.635 '''

''' Function to call Best Model '''
model = supervised._AdaBoost()['model']
joblib.dump(model, path + 'AdaBoostClassifier.sav')
joblib.dump(modeldata.columns, path + 'FeatureNames.sav')

'''   Validating Model on Test Data  '''
testfile = path+'test.csv'

testdata = FR.FileReader(testfile).data
print(testdata.head())

## Transforming Categorical Variables into dummy variable using binning
test_dummy = pd.DataFrame(DTS.DataTransformation().elementBucket(testdata['text']))
print(test_dummy.shape)

## Shaping the data to use for prediction
train_features = joblib.load(path + 'FeatureNames.sav')
diffcolumns = np.setdiff1d(train_features, test_dummy.columns)
diffcols_zeros = pd.DataFrame(np.zeros((test_dummy.shape[0], 
                                        len(diffcolumns))), columns=[diffcolumns])

pr_data = pd.concat([test_dummy, diffcols_zeros], axis = 1)
dropcolumns = np.setdiff1d(test_dummy.columns, train_features)
test_new = pr_data.drop(dropcolumns,axis = 1, inplace = False)

model = joblib.load(path + 'AdaBoostClassifier.sav')
test_pred = model.predict(test_new)
valfile = pd.concat([testdata['id'], pd.DataFrame(test_pred, columns = ['category'])], axis = 1)
valfile.to_csv(path + 'submission.csv',index = None)

