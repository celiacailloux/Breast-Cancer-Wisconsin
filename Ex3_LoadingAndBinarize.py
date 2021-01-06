#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:12:21 2018

@author: celiacailloux

This scripts generates 


"""
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
from similarity import binarize2
from writeapriorifile import WriteAprioriFile
from LoadingDataShort import *

print(X.shape)
print(type(X))
print(attributeNames)
'''
# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]
attributeNames = [name[0][0] for name in mat_data['attributeNames']]
classNames = [cls[0][0] for cls in mat_data['classNames']]
'''
'''
X.shape
Out[20]: (6497, 12)
type(X)
Out[21]: numpy.ndarray

X.shape
Out[14]: (569, 30)
type(X)
Out[15]: numpy.ndarray
'''
X = np.array(X)

binData = binarize2(X,attributeNames)
WriteAprioriFile(binData[0], titles = binData[1])