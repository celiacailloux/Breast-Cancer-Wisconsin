# Exercise 4.1.4
from matplotlib.pyplot import (figure, subplot, boxplot, title, xticks, ylim, 
                               show)
# requires data from exercise 4.1.1
from LoadingData import *

X = np.asarray(X)
y = yBoxPlot

figure(figsize=(14,7))
for c in range(C):
    #print(c+1)
    subplot(1,C,c+1)
    class_mask = (y==c) # binary mask to extract elements of class c
    #print(class_mask)
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c
    
    boxplot(X[class_mask,:])
    #title('Class: {0}'.format(classNames[c]))
    title('Class: '+classNames[c])
    #xticks(range(1,len(attributeNames)+1), [a[:7] for a in attributeNames], rotation=45)
    y_up = X.max()+(X.max()-X.min())*0.1; y_down = X.min()-(X.max()-X.min())*0.1
    ylim(y_down, y_up)

#show()
plt.savefig('boxperclass.png')

print('Ran Exercise 4.1.4')