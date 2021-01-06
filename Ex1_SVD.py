# exercise 2.1.3
# (requires data structures from ex. 2.2.1)
from LoadingData import *

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show
from scipy.linalg import svd

# Subtract mean value from data + standardize by diving with standard deviation
#Y = (X - np.ones((N,1))*X.mean(0))
Y = (X - np.ones((N,1))*X.mean(0))/(np.ones((N,1))*X.std(ddof=1))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

#print(S)
#print(V)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Plot variance explained
figure()
plot(range(1,len(rho)+1),rho,'o-')
#title('Variance Explained by Principal Components');
xlabel('Principal component',fontsize=14);
ylabel('Variance explained',fontsize=14);
#plt.xlabel('xlabel', fontsize=18)
#plt.ylabel('ylabel', fontsize=16)

#plt.savefig('PC.png')
show()



print('Ran Exercise 2.1.3')