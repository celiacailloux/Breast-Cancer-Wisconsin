# exercise 6.3.1
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show, plot, legend, ylim,title
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from scipy import stats
import neurolab as nl
from LoadingData import *
from sklearn.neighbors import KNeighborsClassifier
'''
# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine2.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)
'''
# Loading data
#X = stats.zscore(X)

#y = stats.zscore(y)
#y = np.asarray(y)[:,0]



## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)
#CV = model_selection.StratifiedKFold(n_splits=K)

# Initialize variables
#Error_logreg = np.empty((K,1))
#Error_dectree = np.empty((K,1))

Error_KNN = np.empty((K,1)) #no feature selection
Error_logreg = np.empty((K,1)) #with feature selection
Error_ANN = np.empty((K,1))    #ANN

y_mean_error = np.empty((K,1)) 
    
n_tested=0

# -------------------------------- feature selection
ff = np.array([1, 4, 5, 6])
# --------------------------------
n_train = 5             # number of networks trained in each k-fold
learning_goal = 100     # stop criterion 1 (train mse to be reached) #error
max_epochs = 100#64         # stop criterion 2 (max epochs in training)
show_error_freq = 5 
hidden_units = 2


k=0
for train_index, test_index in CV.split(X,y):
    print('CV-fold {0} of {1}'.format(k+1,K))
    

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    y_mean = np.ones((len(y_test),1))*np.max(y_train)
    y_mean_error[k] =np.square(y_test-y_mean).sum()/y_test.shape[0]
    print(y_mean_error[k])
    
    #------------------------------------------
    #KNN
    y_trainKNN = np.asarray(y_train).T[0]
    y_testKNN = np.asarray(y_test).T[0]
    # K-nearest neighbors
    n_neigbours = 4    
    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=2    
    # Fit classifier and classify the test points
    knclassifier = KNeighborsClassifier(n_neighbors=n_neigbours, p=dist);
    knclassifier.fit(X_train, y_trainKNN);
    y_est = knclassifier.predict(X_test);
    Error_KNN[k] = sum(np.abs(y_est - y_testKNN)) / float(len(y_est))
    
    #------------------------------------------
    # Fit and evaluate Logistic Regression classifier
    model = lm.logistic.LogisticRegression(C=N)
    model = model.fit(X_train, y_train)
    
    y_logreg = model.predict(X_test)    
    y_est = y_logreg
    y_testArray = np.asarray(y_test)[:,0]
    #Error_logreg[k] = 100*(y_logreg!=y_testArray).sum().astype(float)/len(y_testArray)
    
    
    #test_error[k] = (y_est!=y_testArray).sum().astype(float)/y_testArray.shape[0]
    Error_logreg[k] = sum(np.abs(y_est - y_testArray)) / float(len(y_est))
    #------------------------------------------
    
    ann = nl.net.newff([[-3, 3]]*M, [hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])    
    Error_ANN_train = ann.train(X_train, y_trainKNN.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
    y_est = ann.sim(X_test).squeeze()
    Error_ANN[k] = np.power(y_est-y_testKNN,2).sum().astype(float)/y_test.shape[0]  
    
    k+=1

#yp_linreg_avg = [sum(items) / len(yp_linreg) for items in zip(*yp_linreg)]
#print(yp_linreg_avg)
#yp_linreg_fs_avg = np.average(yp_linreg_fs)
#yp_ANN_avg = np.average(yp_ANN)
#print(Error_linreg)
#print(Error_linreg_fs)
#print(Error_ANN)


#-------------------------------------------------
# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 
#z = (Error_logreg-Error_dectree)
print('Error_logreg-Error_ANN')
z = (Error_logreg-Error_ANN)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);
print(sig)
#print(zH)

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')

#-------------------------------------------------
# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 
#z = (Error_logreg-Error_dectree)
print('Error_KNN-Error_logreg')
z = (Error_KNN-Error_logreg)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);
print(sig)
#print(zH)

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')

#-------------------------------------------------
# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 
#z = (Error_logreg-Error_dectree)
print('Error_KNN-Error_ANN')
z = (Error_KNN-Error_ANN)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);
print(sig)

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')

# Boxplot to compare classifier error distributions
print(y_mean_error)
figure()
#boxplot(np.concatenate((Error_logreg, Error_dectree),axis=1))
boxplot(np.concatenate((Error_KNN, Error_logreg, Error_ANN, y_mean_error),axis=1))
xlabel('KNN                Log. Reg                  ANN               Largest y_train')
ylabel('Cross-validation error [%]')

show()



print('Ran Exercise 6.3.1')