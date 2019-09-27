#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:54:17 2019

@author: bibhuti

Desc : CLass to transform data into structured format
"""
import numpy as np

class DataTransformation(object):
    
    def elementFreq(self, data):
        ''' 
            Function takes data as list or array type and returns frequencies as 
            Dictiary with key as element and value as frequencies
        '''
        str_occur_dict = {el : 0 for el in np.unique(data)}
        for val in str_occur_dict.keys():
            str_occur_dict[val] = sum(map(lambda x : x == val, data))
        return str_occur_dict   
    
    def elementBucket(self, data):
        ''' 
        Implementation of One hot encoding Transformation
        '''
        ''' 
        Function takes data as list or array or dataframe series.
        Create frequency bucket and returns as dictinary  
        
        '''
        
        unique_token = np.unique(sum(list(map(lambda x : x.split(' '),data)), [])) ## Extracting all categries in n variab;e
        bucket = []
        str_split = list(map(lambda x : x.split(' ') ,data))
        
        for val in str_split:
            dict_sample = {element:0 for element in unique_token}
            sam1 = self.elementFreq(val)
            for key in sam1.keys():
                dict_sample[key] = sam1[key]
            bucket.append(dict_sample)
        return bucket    

  
        
        
        
        