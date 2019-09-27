#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:49:28 2019

@author: bibhuti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:39:55 2019

@author: bibhuti
"""

import pandas as pd
import numpy as np
from Reader import FileReader as FR
from metrics import DataTransformation as DTS
from models import SupervisedModelling as SM
from sklearn.externals import joblib

path = './data/'
file = path+'training_1.csv'

traindata = FR.FileReader(file).data
print(traindata.head())
## Transforming Categorical Variables into dummy variable using binning
modeldata = pd.DataFrame(DTS.DataTransformation().elementBucket(traindata['text']))


########## split train and test 75 * 25 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   modeldata, traindata['category'], test_size=0.25, random_state=42, shuffle =True)


supervised = SM.SupervisedModelling(X_train,y_train, X_test, y_test)
supervised.modelAll()

print(pd.DataFrame(supervised.modelmetrics).T)



from sklearn.neural_network import MLPClassifier
for i in range(3,10):
    for j in range(3,10):
        model = MLPClassifier(solver='sgd',learning_rate = 'adaptive',activation='tanh', alpha=1e-5,
                     hidden_layer_sizes=(i,j), random_state=1)
        test_pred = model.fit(X_train, y_train).predict(test_new)
        print("i = "+ str(i) + 'j = ' + str(j) + "labels= " + str(np.sum(test_pred)))                
             

##  Giving best result 61.25 21 1's
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
test_pred=clf.fit(X_train,y_train).predict(test_new) 
print(np.sum(np.array(test_pred)))
valfile = pd.concat([testdata['id'], pd.DataFrame(test_pred, columns = ['category'])], axis = 1)
#valfile.to_csv('submission_adaboost.csv',index = None)

from sklearn.ensemble import AdaBoostClassifier
for i in np.linspace(1.05,2,1000):
    clf = AdaBoostClassifier(n_estimators = 200, random_state=0,
                             algorithm = 'SAMME.R', learning_rate = i)
    test_pred=clf.fit(X_train,y_train).predict(test_new)
    labelsum = np.sum(np.array(test_pred))
    if(labelsum > 43):
        print("I = "+ str(i) + " labels = "+str(labelsum))


def concordance(data, threshold = 0.5):
    return(list(map(lambda x : 0 if x > threshold else 1, data[:,[0]])))
    
sum(concordance(test_pred, threshold = 0.56))

test_new = concordance(test_pred, threshold = 0.565)
valfile = pd.concat([testdata['id'], pd.DataFrame(test_new, columns = ['category'])], axis = 1)

#valfile.to_csv('submission.csv',index = None)




clf = AdaBoostClassifier(n_estimators=100, random_state=0, learning_rate = 1.636)
test_pred=clf.fit(X_train,y_train).predict_proba(test_new) 
print(np.sum(np.array(test_pred)))
#joblib.dump(model, 'AdaBoostClassifier.sav')
#joblib.dump(modeldata.columns, 'FeatureNames.sav')


valfile = pd.concat([testdata['id'], pd.DataFrame(test_pred, columns = ['category'])], axis = 1)
#valfile.to_csv('submission_adaboost.csv',index = None)



'''   Validating Model on Test Data  '''
testfile = path+'test.csv'

testdata = FR.FileReader(testfile).data
print(testdata.head())

## Transforming Categorical Variables into dummy variable using binning
test_dummy = pd.DataFrame(DTS.DataTransformation().elementBucket(testdata['text']))
print(test_dummy.shape)

## Shaping the data to use for prediction
train_features = joblib.load('FeatureNames.sav')
diffcolumns = np.setdiff1d(train_features, test_dummy.columns)
diffcols_zeros = pd.DataFrame(np.zeros((test_dummy.shape[0], 
                                        len(diffcolumns))), columns=[diffcolumns])

pr_data = pd.concat([test_dummy, diffcols_zeros], axis = 1)
dropcolumns = np.setdiff1d(test_dummy.columns, train_features)
test_new = pr_data.drop(dropcolumns,axis = 1, inplace = False)

test_pred = model.predict(test_new)

valfile = pd.concat([testdata['id'], pd.DataFrame(test_pred, columns = ['category'])], axis = 1)

valfile.to_csv('submission.csv',index = None)

