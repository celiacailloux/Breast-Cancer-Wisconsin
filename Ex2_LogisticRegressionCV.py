# exercise 6.3.1

from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show, plot, legend, ylim,title
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from scipy import stats
from LoadingData import *
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
## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)
#CV = model_selection.StratifiedKFold(n_splits=K)

# Initialize variables
Error_logreg = np.empty((K,1))
misclass_rate = np.empty((K,1))
Error_dectree = np.empty((K,1))
test_error = np.empty((K,1))
n_tested=0

k=0
for train_index, test_index in CV.split(X,y):
    print('CV-fold {0} of {1}'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit and evaluate Logistic Regression classifier
    model = lm.logistic.LogisticRegression(C=N)
    model = model.fit(X_train, y_train)
    
    y_logreg = model.predict(X_test)
    
    
    y_est = y_logreg
    y_testArray = np.asarray(y_test)[:,0]
    #Error_logreg[k] = 100*(y_logreg!=y_testArray).sum().astype(float)/len(y_testArray)
    
    
    #test_error[k] = (y_est!=y_testArray).sum().astype(float)/y_testArray.shape[0]
    misclass_rate[k] = sum(np.abs(y_est - y_testArray)) / float(len(y_est))
    
    
    #------------------------------------------
    y_est_white_prob = model.predict_proba(X_test)[:, 0] 
    
    f = figure();
    # np.nonzero returns the elements that are nonzero
    class0_ids = np.nonzero(y_test==0)[0].tolist()
    class1_ids = np.nonzero(y_test==1)[0].tolist()
    
    plot(class0_ids, y_est_white_prob[class0_ids], '.y')
    plot(class1_ids, y_est_white_prob[class1_ids], '.r')
    xlabel('Data object (wine samples $x_i$)'); ylabel('Predicted prob. of class White');
    title('fold {0}'.format(k+1))
    #legend(['White', 'Red'])
    legend([classNames[0],classNames[1]])#['White', 'Red'])
    ylim(-0.01,1.5)
    
    show()
    
    #------------------------------------------
    '''
    # Fit and evaluate Decision Tree classifier
    model2 = tree.DecisionTreeClassifier()
    model2 = model2.fit(X_train, y_train)
    y_dectree = model2.predict(X_test)
    Error_dectree[k] = 100*(y_dectree!=y_test).sum().astype(float)/len(y_test)
    '''
    k+=1
#print(Error_logreg)

#print(test_error)
print('Testerror ', misclass_rate)

'''
# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 
z = (Error_logreg-Error_dectree)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_logreg, Error_dectree),axis=1))
xlabel('Logistic Regression   vs.   Decision Tree')
ylabel('Cross-validation error [%]')

show()
'''

print('Ran Exercise 6.3.1')