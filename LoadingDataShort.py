#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:13:24 2018

@author: celiacailloux
"""
import xlrd
import numpy as np


#Load xls sheet with data 'Users/osezeiyore/Dropbox/02450\ Machine \ Learning/EgneScripts/AttributeNames.xlsx'
#docNames = xlrd.open_workbook('/celiacailloux/Dropbox/02450 Machine Learning/EgneScripts/AttributeNames.xlsx').sheet_by_index(0)
docNames = xlrd.open_workbook('/Users/celiacailloux/Dropbox/02450 Machine Learning/EgneScripts_celia/AttributeNames.xlsx').sheet_by_index(0)
doc = xlrd.open_workbook('/Users/celiacailloux/Dropbox/02450 Machine Learning/EgneScripts_celia/WDBC.xls').sheet_by_index(0)

# 1st column is ID
# 2nd column is classLabels
# Extract attribute names (1st row, column 3 to 32)
c_i = 2     # inital column index for attributes
#c_f = 32    # final column index for attribues
c_f = 12    # final column index for attribues
#attributeNames = doc.row_values(0, 2, 32)
#attributeNames = doc.row_values(0, 2, 32)
attributeNames = docNames.row_values(0, c_i, c_f)



# Extract class names to python list,
# then encode with integers (dict)
r_i = 1     # inital row index for observations
r_f = 570   # final row index for observations
classLabels = doc.col_values(1, r_i, r_f)
classNames = sorted(set(classLabels))

classDict = dict(zip(classNames, range(len(classNames))))

#print('AttributeNames: ', attributeNames)
#print('ClassLabels: ', classLabels)
#print('ClassNames: ', classNames)
#print('Encoded ClassNames: ', classDict)

# Extract vector y, convert to NumPy matrix and transpose
y = np.mat([classDict[value] for value in classLabels]).T

# The following is for regressionx
col_reg = 5 #column chosen for regression
y_reg =np.mat(doc.col_values(col_reg, r_i, r_f)).T

yBoxPlot = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract excel data to matrix X
# there are 569 observations and 30 attributes
n_obs = 569 # number of observations
n_att = c_f-c_i  # number of attributes

X = np.mat(np.empty((n_obs, n_att)))
for i, col_id in enumerate(range(c_i, c_f)):
    X[:, i] = np.mat(doc.col_values(col_id, r_i, r_f)).T
    
    
N = len(y)
M = len(attributeNames)
C = len(classNames)

#print(N, M, C)

data_title = 'Wisconsin Diagnostic Breast Cancer (WDBC)'
sh_path = '/Users/celiacailloux/Dropbox/Apps/ShareLaTeX/02450 - Assignment 3/fig/'#sharelatex

#print('Vector y (the transposed is printed): \n', y.T)




    