# Exercise 4.1.7

from matplotlib.pyplot import (figure, imshow, xticks, xlabel, ylabel, title, 
                               colorbar, cm, show)
from scipy.stats import zscore

# requires data from exercise 4.1.1
from LoadingData import *

X_standarized = zscore(X, ddof=1)

#maybe aspect should be 2./N?
figure(figsize=(4*12,3*6))
imshow(X_standarized, interpolation='none', aspect=(4./N), cmap=cm.gray);
xticks(range(len(attributeNames)), attributeNames)
xlabel('Attributes')
ylabel('Data objects')
title(data_title)
colorbar()

show()

print('Ran Exercise 4.1.7')