#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:21:14 2019

@author: bibhuti

Desc : Class to read different file types
"""

import pandas as pd

class FileReader(object):
    
    def __init__(self, pathWithFile, filetype = 'csv'):
        self.__file = pathWithFile
        self.data = None
        self.__csvReader()
        
        
    def __csvReader(self):
        self.data = pd.read_csv(self.__file)
    

#file = '/Users/bibhu/Downloads/Data Semantics/training_1.csv'
#reader = FileReader(file)

    