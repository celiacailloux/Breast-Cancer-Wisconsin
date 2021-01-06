# exercise 11.1.1
from matplotlib.pyplot import figure, show, savefig, ylabel, xlabel
import numpy as np
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.mixture import GaussianMixture
from LoadingDataD import *
import Ex3_PCA 
'''
# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/synth1.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
#X_old = X
#X = np.hstack([X,X])
N, M = X.shape
C = len(classNames)

# ---- types
type(X): numpy.ndarray
X.shape: (200, 2)
type(y): numpy.ndarray

# ----- type 

X.shape: (200, 2)
cls.shape: (200,)
cds.shape: (10, 2)
(10, 2, 2)
'''

y = np.asarray(y)[:,0]

# Number of clusters
K = 2

# ---------------------------------------- COVARIANCE
cov_type = 'full'       
# type of covariance, you can try out 'diag' as well

reps = 3               
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model

''' NON PCA
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(X)

cls = gmm.predict(X)
# extract cluster labels
cds = gmm.means_  
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_



# extract cluster shapes (covariances of gaussians)

if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
figure(figsize=(14,9))
clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
show()
'''

# PCA

X_PCA = Ex3_PCA.PCA(X)
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(X_PCA)
cls_PCA = gmm.predict(X_PCA)
cds_PCA = gmm.means_  
covs_PCA =   gmm.covariances_

# extract cluster shapes (covariances of gaussians)

if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
figure(figsize=(14,9))
clusterplot(X_PCA, clusterid=cls_PCA, centroids=cds_PCA, y=y, covars=covs_PCA)
xlabel('PCA1', fontsize = 18)
ylabel('PCA2', fontsize = 18)

## In case the number of features != 2, then a subset of features most be plotted instead.
#figure(figsize=(14,9))
#idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
#clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
#show()

sh_path = '/Users/celiacailloux/Dropbox/Apps/ShareLaTeX/02450 - Assignment 3/fig/'#sharelatex
figname = sh_path + 'PCAplot_K=' + str(K) + '.png'
savefig(figname)
show()
print('Ran Exercise 11.1.1')