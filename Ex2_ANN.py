# exercise 8.2.6

from matplotlib.pyplot import figure, plot, subplot, title, show, bar
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection
from scipy import stats
from Ex2_LoadingDataReg import *

'''
# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine2.mat')
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
X = mat_data['X']
y = X[:,10]             # alcohol contents (target)
X = X[:,1:10]           # the rest of features
N, M = X.shape
C = 2
'''
# Normalize data
X = stats.zscore(X)
y = np.asarray(y_reg)[:,0]
y = stats.zscore(y)
                
## Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
#Y = stats.zscore(X,0);
#U,S,V = np.linalg.svd(Y,full_matrices=False)
#V = V.T
##Components to be included as features
#k_pca = 3
#X = X @ V[:,0:k_pca]
#N, M = X.shape


# Parameters for neural network classifier

hidden_units = np.linspace(2,10,5, dtype = int)
#hidden_units = np.linspace(1,10,5, dtype = int)
n_train = len(hidden_units)             # number of networks trained in each k-fold
learning_goal = 50#2.0     # stop criterion 1 (train mse to be reached) #error
max_epochs = 64         # stop criterion 2 (max epochs in training)
show_error_freq = 20   # frequency of training status updates

# K-fold crossvalidation
KOuter = 5#3                   # only three folds to speed up this example
KInner = 10#4

CVOuter = model_selection.KFold(KOuter,shuffle=True)
CVInner = model_selection.KFold(KInner,shuffle=True)

'''
# Variable for classification error
errors = np.zeros(K)*np.nan
error_hist = np.zeros((max_epochs,K))*np.nan
bestnet = list()
'''

testerrormatrix = np.zeros([KInner,n_train])
test_error_best = np.zeros(KOuter)
best_n_hiddens = np.zeros(KOuter)
kOuter_it = 0
for train_index, test_index in CVOuter.split(X,y):
    print('\nOuter Crossvalidation fold: {0}/{1}'.format(kOuter_it+1,KOuter))   
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    
    kInner_it = 0 
    #n_hidden_units = 2      # number of hidden units     
    for train_index2, test_index2 in CVInner.split(X_train,y_train):
        print('\n - Inner Crossvalidation fold: {0}/{1}'.format(kInner_it+1,KInner))   
        
        # extract training and test set for current CV fold
        X_train2 = X_train[train_index2,:]
        y_train2 = y_train[train_index2]
        X_test2 = X_train[test_index2,:]
        y_test2 = y_train[test_index2]
        
        #best_train_error = np.inf
        #In this for loop we try five different number of hidden units
        #For each we train and then test and save the test error
        for i in range(n_train):
            print('\n -- Training network {0}/{1}...'.format(i+1,n_train))
            # Create randomly initialized network with 2 layers
            ann = nl.net.newff([[-3, 3]]*M, [hidden_units[i], 1], [nl.trans.TanSig(),nl.trans.PureLin()])
            
            train_error = ann.train(X_train2, y_train2.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)            
            y_est = ann.sim(X_test2).squeeze()
            test_error = np.power(y_est-y_test2,2).sum().astype(float)/y_test2.shape[0]
            testerrormatrix[kInner_it,i] = test_error   
            #print('Number of hidden units', hidden_units[i])
            #print(test_error)
            #print(kInner_it,i)
            
            
            '''
            if i==0:
                bestnet.append(ann)
            # train network
            train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
            if train_error[-1]<best_train_error:
                bestnet[k]=ann
                best_train_error = train_error[-1]
                error_hist[range(len(train_error)),k] = train_error
            '''
            
        kInner_it +=1
        
        '''
        print('Best train error: {0}...'.format(best_train_error))
        y_est = bestnet[k].sim(X_test).squeeze()
        errors[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
        '''
        # why the power of y_est-y_test?2
    print('{0}'.format(np.round(testerrormatrix,2)))
    test_error_mean = np.mean(testerrormatrix, axis=0)
    print('{0}'.format(np.round(test_error_mean,2)))
    best_n_hidden = hidden_units[np.argmin(test_error_mean)]
    best_n_hiddens[kOuter_it] = best_n_hidden
    
    
    
    ann_best = nl.net.newff([[-3, 3]]*M, [best_n_hidden, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
    train_error_best = ann_best.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
    
    y_est = ann.sim(X_test).squeeze()
    test_error = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    test_error_best[kOuter_it] = test_error
        
    #print('Best train error: {0}...'.format(best_train_error))
    #y_est = bestnet[k].sim(X_test).squeeze()
    #errors[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    kOuter_it +=1
print('\n*************************')
print(best_n_hiddens)
print(test_error_best)
print('The number of hidden units that given the lowest test_error is: {0:.3f}'.format(best_n_hiddens[np.argmin(test_error_best)]))

'''

# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(errors)))

figure(figsize=(6,8));

subplot(2,1,1); bar(range(0,K),errors); title('Mean-square errors');

subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');

figure(figsize=(6,7));
subplot(2,1,1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y'); 
subplot(2,1,2); plot((y_est-y_test)); title('Last CV-fold: prediction error (est_y-test_y)'); 
show()

print('Ran Exercise 8.2.6')

#% The weights if the network can be extracted via
#bestnet[0].layers[0].np['w'] # Get the weights of the first layer
#bestnet[0].layers[0].np['b'] # Get the bias of the first layer
'''
