# exercise 2.2.4
from LoadingData import *
from scipy.linalg import svd
from pprint import pprint

'''This script will plot the attribute names together with a principal
component. PCAindex (0, 1, 2 ..) will determine which PCA the attributenames
are plottet together with.
'''

PCAindex = 1 # 0 = PCA1, 1 = PCA2, ... etc

# (requires data structures from ex. 2.2. and 2.2.3)
np.set_printoptions(precision=2) # sets print options to show two decimals
Y = X - np.ones((N,1))*X.mean(0)
U,S,V = svd(Y,full_matrices=False)
V=V.T




#print(V[:,1].T)
tempAttributeNamesPCA2 = dict(zip(attributeNames, V[:,1].T))
pprint(tempAttributeNamesPCA2)
#print(np.round(V[:,1].T,2))

## Projection of water class onto the 2nd principal component.
# Note Y is a numpy matrix, while V is a numpy array. 

print('\n')

classIndex = [0,1] # 'B' or 'M'
# Either convert V to a numpy.mat and use * (matrix multiplication)
print((Y[y.A.ravel()==classIndex[0],:] * np.mat(V[:,1]).T).T)

# Or interpret Y as a numpy.array and use @ (matrix multiplication for np.array)
#print( (np.asarray(Y[y.A.ravel()==4,:]) @ V[:,1]).T )

print('Ran Exercise 2.1.5')