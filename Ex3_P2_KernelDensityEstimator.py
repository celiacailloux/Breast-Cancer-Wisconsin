# exercise 11.2.2
import numpy as np
from matplotlib.pyplot import figure, subplot, hist, title, show, plot
from scipy.stats.kde import gaussian_kde
from scipy import stats
from LoadingData import *

'''
The script estimates the density using a kernel density estimator with a 
Gaussian kernel and a kernel width of 1, and plot the density on the 
range −10 to 10.
'''
X = stats.zscore(X)
'''
# Draw samples from mixture of gaussians (as in exercise 11.1.1)
N = 1000; M = 1
x = np.linspace(-10, 10, 50)
X = np.empty((N,M))
'''

#mean an covariance
m = np.array([1, 3, 6]);
s = np.array([1, .5, 2])
c_sizes = np.random.multinomial(N, [1./3, 1./3, 1./3])

for c_id, c_size in enumerate(c_sizes):
    X[c_sizes.cumsum()[c_id]-c_sizes[c_id]:c_sizes.cumsum()[c_id],:] = np.random.normal(m[c_id], np.sqrt(s[c_id]), (c_size,M))


# x-values to evaluate the KDE
xe = np.linspace(-10, 10, 100)

# Compute kernel density estimate
kde = gaussian_kde(X.ravel())

# Plot kernel density estimate
figure(figsize=(6,7))
subplot(2,1,1)
hist(X,x)
title('Data histogram')
subplot(2,1,2)
plot(xe, kde.evaluate(xe))
title('Kernel density estimate')
show()

print('Ran Exercise 11.2.2')