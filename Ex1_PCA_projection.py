# exercise 2.2.4
from LoadingData import *
from scipy.linalg import svd
from pprint import pprint
#import plotly
#import plotly.plotly as py
#import plotly.graph_objs as go


'''This script will plot the attribute names together with a principal
component. PCAindex (0, 1, 2 ..) will determine which PCA the attributenames
are plottet together with. classIndex will determine 
'''

PCAindex = 1 # 0 = PCA1, 1 = PCA2, ... etc
PCAindex0 = 0
classIndex = 0 # 0 = 'B' and 1 = 'M'

# (requires data structures from ex. 2.2. and 2.2.3)
np.set_printoptions(precision=2) # sets print options to show two decimals
Y = X - np.ones((N,1))*X.mean(0)
U,S,V = svd(Y,full_matrices=False)
V=V.T


#print(V[:,1].T)
tempAttributeNamesPCA2 = dict(zip(attributeNames, V[:,PCAindex].T))
print('----------------------------------------------------------------------')
print('Printing the attribute names together witn PCA', PCAindex+1 ,'.')
print('From this we should be able to determine which of the attributes that')
print('PCA', PCAindex+1, ' mainly captures the variation of (which numbers are big?).')
print('\n')
pprint(tempAttributeNamesPCA2)
print('\n')
#print(np.round(V[:,1].T,2))



#print(V[:,1].T)
tempAttributeNamesPCA1 = dict(zip(attributeNames, V[:,PCAindex0].T))
print('----------------------------------------------------------------------')
print('Printing the attribute names together witn PCA', PCAindex0+1 ,'.')
print('From this we should be able to determine which of the attributes that')
print('PCA', PCAindex0+1, ' mainly captures the variation of (which numbers are big?).')
print('\n')
pprint(tempAttributeNamesPCA1)
print('\n')
#print(np.round(V[:,1].T,2))



## Projection of className onto the principal component PCAx.
# Note Y is a numpy matrix, while V is a numpy array. 

# Either convert V to a numpy.mat and use * (matrix multiplication)
print('----------------------------------------------------------------------')
print('Projection of ', classNames[classIndex], ' onto the PCA', PCAindex+1,'.')
print('What would cause an observation to have a large negative/positive')
print('projection onto PCA', PCAindex+1)
print('\n')
print((Y[y.A.ravel()==classIndex,:] * np.mat(V[:,1]).T).T)

# Or interpret Y as a numpy.array and use @ (matrix multiplication for np.array)
#print( (np.asarray(Y[y.A.ravel()==4,:]) @ V[:,1]).T )

print('Ran Exercise 2.1.5')

print('V[:,PCAindex0].T')

#table1=[attributeNames, V[:,PCAindex].T,  
#print('table1')

#trace = go.Table(
##    header=dict(values=['A Scores', 'B Scores','c']),
#    cells=dict(values=[attributeNames,
#                       [V[:,PCAindex].T],[V[:,PCAindex0].T]]))
#
#data = [trace] 
#py.iplot(data, filename = 'basic_table')

##
#plt.plot(V[:,PCAindex0].T,V[:,PCAindex].T,'o')
#plt.show()