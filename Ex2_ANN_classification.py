# exercise 8.2.2
# ----------------------------------- ANN
# X has the shape (400,2) thus only two attributes.

from matplotlib.pyplot import (figure, plot, subplot, title, xlabel, ylabel, 
                               hold, contour, contourf, cm, colorbar, show,
                               legend, savefig, legend)
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection
from LoadingData import *
# read XOR DATA from matlab datafile
# ---------------------------------- SIMPLY READING DATA
'''
mat_data = loadmat('../Data/xor.mat')
X = mat_data['X']
y = mat_data['y']

attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0] for name in mat_data['classNames'].squeeze()]
N, M = X.shape
C = len(classNames)
'''
X = stats.zscore(X); #normalizing data
# ---------------------------------- ANN parameters
# Parameters for neural network classifier        
hidden_units = np.linspace(2,10,5, dtype = int) # number of hidden units
n_train = 5                # number of networks trained in each k-fold

# These parameters are usually adjusted to: 
# (1) data specifics, 
# (2) computational constraints
learning_goal = 100#2.0     # stop criterion 1 (train mse to be reached)
max_epochs = 100#200        # stop criterion 2 (max epochs in training)
show_error_freq = 5 

# ---------------------------------- K-fold CV
# K-fold CrossValidation 
# (4 folds here to speed up this example)
K = 4 #Kouter
KInner = 4#4
CV = model_selection.KFold(K,shuffle=True)
CVInner = model_selection.KFold(KInner,shuffle=True)

test_error_matrix = np.zeros([KInner,n_train])
test_error_best = np.zeros(K)
best_n_hiddens = np.zeros(K)
# Variable for classification error
#errors = np.zeros(K)*np.nan
#error_hist = np.zeros((max_epochs,K))*np.nan #we dont't need train errors
#bestnet = list()


k=0
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    

    # ---------------------------------- Defining the train and test set    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index,:]
    X_test = X[test_index,:]
    y_test = y[test_index,:]
    
    kInner_it = 0 
    n_hidden_units = 2  
    
    for train_index2, test_index2 in CVInner.split(X_train,y_train):
        print('\n - Inner Crossvalidation fold: {0}/{1}'.format(kInner_it+1,KInner)) 
        
                # extract training and test set for current CV fold
        X_train2 = X_train[train_index2,:]
        y_train2 = y_train[train_index2]
        X_test2 = X_train[test_index2,:]
        y_test2 = y_train[test_index2]
    
        # ---------------------------------- Run ANN n_train times
        #best_train_error = 1e100
        for i in range(n_train):
            print('\nOuter Crossvalidation fold: {0}/{1}'.format(k+1,K))
            # ---------------------------------- ANN
            # Create randomly initialized network with 2 layers
            #ann = nl.net.newff([[0, 1], [0, 1]], [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
            ann = nl.net.newff([[0, 1]]*M, [hidden_units[i], 1], [nl.trans.TanSig(),nl.trans.PureLin()])
            # train network
            # train error is not used!
            train_error = ann.train(X_train2, y_train2, goal=learning_goal, epochs=max_epochs, show=round(max_epochs/8))
            
            # ---------------------------------- Continue the 
            #if train_error[-1]<best_train_error:
            #    bestnet.append(ann)
            #    best_train_error = train_error[-1]
            #    error_hist[range(len(train_error)),k] = train_error
            y_est = ann.sim(X_test2)
            y_est = (y_est>.5).astype(int)     # convert list with boolean strings (True/False) with 0 and 1
            #train error
            test_error = (y_est!=y_test2).sum().astype(float)/y_test2.shape[0]
            test_error_matrix[kInner_it,i] = test_error
            
        # ---------------------------------- Defining the train and test set
        kInner_it +=1
    test_error_mean = np.mean(test_error_matrix, axis=0)
    best_n_hidden = hidden_units[np.argmin(test_error_mean)]
    best_n_hiddens[k] = best_n_hidden
        
    ann_best = nl.net.newff([[0, 1]]*M, [best_n_hidden, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
    train_error_best = ann_best.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
    
    y_est = ann.sim(X_test)
    y_est = (y_est>.5).astype(int)
    test_error_best[k] = (y_est!=y_test).sum().astype(float)/y_test.shape[0]
    k+=1
    
# Print the average classification error rate
#print('Average classification Error rate: {0}%'.format(100*np.mean(errors)))
print('Test errors: {0}'.format(test_error_best))
print('The best numbers of hidden units: {0}'.format(best_n_hiddens))

'''
# Display the decision boundary for the several crossvalidation folds.
# (create grid of points, compute network output for each point, color-code and plot).
grid_range = [-1, 2, -1, 2]; delta = 0.05; levels = 100
a = np.arange(grid_range[0],grid_range[1],delta)
b = np.arange(grid_range[2],grid_range[3],delta)
A, B = np.meshgrid(a, b)
values = np.zeros(A.shape)
'''
'''
figure(1,figsize=(18,9))
for k in range(4):
    subplot(2,2,k+1)
    cmask = (y==0).squeeze(); plot(X[cmask,0], X[cmask,1],'.r')
    cmask = (y==1).squeeze(); plot(X[cmask,0], X[cmask,1],'.b')
    title('Model prediction and decision boundary (kfold={0})'.format(k+1))
    xlabel('Feature 1'); ylabel('Feature 2');
    for i in range(len(a)):
        for j in range(len(b)):
            values[i,j] = bestnet[k].sim( np.mat([a[i],b[j]]) )[0,0]
    contour(A, B, values, levels=[.5], colors=['k'], linestyles='dashed')
    contourf(A, B, values, levels=np.linspace(values.min(),values.max(),levels), cmap=cm.RdBu)
    if k==0: colorbar(); legend(['Class A (y=0)', 'Class B (y=1)'])
    
figname1 = 'Model prediction for ' + str(n_hidden_units)
#savefig('/PlotsExercise2/'+ figname1 + '.png')
savefig(figname1 + '.png')
'''
'''
# Display exemplary networks learning curve (best network of each fold)
figure(2)
bn_id = np.argmax(error_hist[-1,:])
error_hist[error_hist==0] = learning_goal
for bn_id in range(K):
    plot(error_hist[:,bn_id], label = 'CV'+ str(bn_id+1))
    xlabel('epoch')
    ylabel('train error (mse)')
    title('Learning curve (best for each CV fold)')
legend()

for bn_id in range(K):
    plot(error_hist[:,bn_id], label = 'CV'+ str(bn_id+1))
    xlabel('epoch')
    ylabel('train error (mse)')
    title('Learning curve (best for each CV fold)')
legend()

plot(range(max_epochs), [learning_goal]*max_epochs, '-.')
figname2 = 'ex8_2_2_LearningCurve' + str(n_hidden_units)
savefig( figname2 + '.png' )

'''

#show()

print('Ran Exercise 8.2.2')