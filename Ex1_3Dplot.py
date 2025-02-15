# Exercise 4.1.6
'''
This script renders a non-interactive 3D plot of three attributes
'''

from matplotlib.pyplot import (figure, show)#, hold)
from mpl_toolkits.mplot3d import Axes3D

# self-made modules
# requires data from exercise 4.1.1
# from LoadingData import *
from data.loading_data import *

X = np.asarray(X)
y = yBoxPlot

# Indices of the variables to plot
ind = [10, 5, 6]
colors = ['blue', 'green', 'red']

f = figure()
ax = f.add_subplot(111, projection='3d') #Here the mpl_toolkits is used
for c in range(C):
    class_mask = (y==c)
    s = ax.scatter(X[class_mask,ind[0]], X[class_mask,ind[1]], X[class_mask,ind[2]], c=colors[c])

ax.view_init(30, 220)
ax.set_xlabel(attributeNames[ind[0]])
ax.set_ylabel(attributeNames[ind[1]])
ax.set_zlabel(attributeNames[ind[2]])

show()

print('Ran Exercise 4.1.6')