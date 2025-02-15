# exercise 6.1.1

from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show
from scipy.io import loadmat
from sklearn import model_selection, tree
import numpy as np

'''
Data: This script is written run on data where outliers has been removed
Data: is divided into test data and training data   
Methods: simple hold-out cross-validation
Output: A graph that shows what the optimal tree depth should be. The optimal 
tree depth is where the test data error has a minimum.
'''

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine2.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1) # a vector from 2..21 (19 numbers in total)

# Simple holdout-set crossvalidation
# Uses some predefined function to split the data
# This predefined function splits the data differently everytime
test_proportion = 0.5
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)

# Initialize variables
Error_train = np.empty((len(tc),1)) #np.empty just makes an array of appropiate size.
Error_test = np.empty((len(tc),1))

# iterates from 2 to 21
for i, t in enumerate(tc):
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
    dtc = dtc.fit(X_train,y_train)

    # Evaluate classifier's misclassification rate over train/test data
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
    misclass_rate_test = sum(np.abs(y_est_test - y_test)) / float(len(y_est_test))
    misclass_rate_train = sum(np.abs(y_est_train - y_train)) / float(len(y_est_train))
    Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train
    
f = figure()
plot(tc, Error_train)
plot(tc, Error_test)
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate)')
legend(['Error_train','Error_test'])
    
show()    

print('Ran Exercise 6.1.1')