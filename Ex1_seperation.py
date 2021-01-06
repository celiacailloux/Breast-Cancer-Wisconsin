#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:14:30 2018

@author: osezeiyore
"""

#from matplotlib.pyplot import boxplot, xticks, ylabel, title, show
import matplotlib.pyplot as plt
# requires data from exercise 4.1.1
from LoadingData import *#ex4_1_1 import *
import numpy as np 

np.where(str(classLabels) == 'M')
indexM = (np.where(np.array(classLabels)== 'M'))
indexB = (np.where(np.array(classLabels)== 'B'))

matM = X[indexM,0:10]
matM = matM[0,:,:]
matB = X[indexB,0:10]
matB = matB[0,:,:]

meanM = np.mean(matM,0)
meanB = np.mean(matB,0)
vstdM = np.std(matM,0)
vstdB = np.std(matB,0)
vvarM = np.var(matM,0)
vvarB = np.var(matB,0)

standM = (matM-meanM)/vstdM
standB = (matB-meanB)/vstdB

#for loop gem disse i vector
#maxM=[1 1 1 1 1 1 1 1 1 1]

for i in range(10):
    #print(np.max(matM[i,:]))
   # print(np.min(matM[:,i]))
   print(np.min(matM[:,i]) , np.max(matM[:,i]))
 #   print(np.min(matM[:,i]) ,  np.max(matM[:,i]))
    #print(np.min(matB[:,i]))




#boxplot
plt.boxplot(np.asarray(standM))
plt.xticks(range(0, 10),attributeNames, rotation = 45, fontsize=12)
plt.show()
#plt.savefig('box1M.png')
plt.boxplot(np.asarray(standB))
plt.xticks(range(0, 10),attributeNames, rotation = 45, fontsize=12)
plt.show()
#plt.savefig('box1B.png')

#plt.boxplot(np.asarray(matM[:,2]))
#plt.show()
plt.savefig('box2M.png')
#plt.boxplot(np.asarray(matB[:,2]))
#plt.show()
plt.savefig('box2B.png')

#-mean /std

#ranM = min max
#ranB = 


#print(np.mean(matM,1))
#print(np.shape(np.mean(matM,1)))
#print(np.shape(X))
    
 #   plt.title(data_title + ' - boxplot')
