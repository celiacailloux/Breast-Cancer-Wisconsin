# exercise 6.2.1
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, savefig
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn import feature_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
from scipy import stats
from tabulate import tabulate


from Ex2_LoadingDataReg import *
y = np.asarray(y_reg)[:,0]
'''
from Ex2_LoadingDataRegAutomated import *
y = np.asarray(y)[:,0]
regType = 'No'
'''
#X = (X - np.ones((N,1))*X.mean(0))/(np.ones((N,1))*X.std(ddof=1)) #Normalizing
X = stats.zscore(X); #Normalizing
y = stats.zscore(y)


## Crossvalidation
# Create crossvalidation partition for evaluation
# Use of cross validation to estimate the performance of our model

K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True) 

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0


for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    #10-fold is used to perform sequential feature selection
    
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]
    
    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    
    # Compute squared error with feature subset selection
    #textout = 'verbose';
    textout = '';
    
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    #Features[selected_features,k]=1
    Features[selected_features,k]=1  
    # .. alternatively you could use module sklearn.feature_selection
    
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        # y_test-y_pred
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration') 

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1
 


# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')
fig_outerCV = 'PlotsExercise2/OuterCrossValidationLoop.png'
savefig(fig_outerCV)

#-------------- New plot


# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual


f=np.argmin(Error_test_fs) # cross-validation fold to inspect
ff=Features[:,f-1].nonzero()[0]
if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
else:
    m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
    
    # -------------------------------------------- Manually determining y_est
    #y_est = model.intercept_ + X @ model.coef_  
    
    y_est= m.predict(X[:,ff])
    residual=y-y_est
    print('- Coefficients are {0} {1}'.format(m.intercept_,m.coef_ ))
    
    figure(k+1, figsize=(12,6))
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       subplot(2,np.ceil(len(ff)/2.0),i+1)
       plot(X[:,ff[i]],residual,'.')
       xlabel(attributeNames[ff[i]])
       ylabel('residual error')

figname = 'PlotsExercise2/reg' + regType + 'TransformFold{0}.png'.format(f)
savefig(figname)
print(Error_test_fs.mean())

    
show()


#ytemp = np.asmatrix(y) # horizontal matrix (1,569)
#Xtemp = X[:,ff] #(569,7)
#par = (np.asmatrix(m.coef_)).T #(7,1)
#print(tabulate([['Alice', 24], ['Bob', 19]], headers=['Name', 'Age']))
#print(tabulate([str(x) for x in attributeNames]))
print('Ran Exercise 6.2.1')