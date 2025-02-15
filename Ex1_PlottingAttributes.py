#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:06:15 2018

@author: celiacailloux
"""

from LoadingData import * 
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# Data attributes to be plotted
i = 20
j = 7

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
#X = np.array(X) #Try to uncomment this line
#plot(X[:, i], X[:, j], 'o')
    
# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
title(data_title)

for c in range(C):
    # select indices belonging to class c:
    class_mask = y.A.ravel()==c
    plot(X[class_mask,i], X[class_mask,j], 'o')

legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()
print('Ran Exercise 2.1.2')