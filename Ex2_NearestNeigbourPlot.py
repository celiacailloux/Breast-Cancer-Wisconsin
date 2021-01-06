# exercise 7.1.1

from matplotlib.pyplot import (figure, hold, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from LoadingData import *
from sklearn import model_selection
import numpy as np
'''

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/synth1.mat')
X = mat_data['X']
X_train = mat_data['X_train']
X_test = mat_data['X_test']
y = mat_data['y'].squeeze()
y_train = mat_data['y_train'].squeeze()
y_test = mat_data['y_test'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)
'''

# Plot the training data points (color-coded) and test data points.
figure(1)

K = 5
CV = model_selection.KFold(K)

for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    y_train = np.asarray(y_train).T[0]
    styles = ['.b', '.r', '.g', '.y']
    for c in range(C):
        class_mask = (y_train==c)
        plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])
    
    
    # K-nearest neighbors
    n_neigbours = 4
    
    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=2
    
    # Fit classifier and classify the test points
    knclassifier = KNeighborsClassifier(n_neighbors=n_neigbours, p=dist);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    
    
    # Plot the classfication results
    styles = ['ob', 'or', 'og', 'oy']
    for c in range(C):
        class_mask = (y_est==c)
        plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
        plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
    title('Synthetic data classification - KNN');
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_est);
    accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    figure(2);
    imshow(cm, cmap='binary', interpolation='None');
    colorbar()
    xticks(range(C)); yticks(range(C));
    xlabel('Predicted class'); ylabel('Actual class');
    title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));
    
    show()
    

print('Ran Exercise 7.1.1')