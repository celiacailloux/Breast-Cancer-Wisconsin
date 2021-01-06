# exercise 2.1.4
# (requires data structures from ex. 2.2.1 and 2.2.3)
from LoadingData import *

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend,scatter
from scipy.linalg import svd


def PCA(X):
    # Subtract mean value from data
    #Y = X - np.ones((N,1))*X.mean(0)
    Y = (X - np.ones((N,1))*X.mean(0))/(np.ones((N,1))*X.std(ddof=1)) #Normalizing
    
    # PCA by computing SVD of Y
    U,S,V = svd(Y,full_matrices=False)
    V = V.T
    # Project the centered data onto principal component space
    Z = Y * V
    
    
    # Indices of the principal components to be plotted
    i = 0
    j = 1
    
    # Plot PCA of the data
    '''
    f = figure()
    '''
    #title(data_title)
    #Z = array(Z)
    
    '''
    
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y.A.ravel()==c
        plot(Z[class_mask,i], Z[class_mask,j], 'o')
        print(Z[class_mask,i].shape)
    #   
    '''
    PCA1 = np.asarray(Z[:,0])[:,0]
    PCA2 = np.asarray(Z[:,1])[:,0]
    #print(PCA1.shape)
    PCA = np.vstack((PCA1, PCA2)).T

    '''
    print(type(PCA))
    print(PCA.shape)
    print(PCA)
    '''
    #print(Z[:,0].shape)
    '''
    
    #scatter(PCA[0], PCA[1])
    #legend(classNames,fontsize=12)
    #legend(['Benign'],['Malignant'],fontsize=12)
    legend(('Benign', 'Malignant'),fontsize=12)
    xlabel('PC{0}'.format(i+1),fontsize=14)
    ylabel('PC{0}'.format(j+1),fontsize=14)
    
    # Output result to screen
    show()
    '''
    print('Found PCA')
    #return np.array(PCA1,PCA2)
    return PCA