# Exercise 4.1.5

from matplotlib.pyplot import (figure, hold, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show)

# requires data from exercise 4.1.1
from LoadingData import *

X = np.asarray(X)
y = yBoxPlot


M=10

##figure(figsize=(10*12,10*10))
#figure(figsize=(17,19))
#for m1 in range(M):
#    for m2 in range(M):
#        subplot(M, M, m1*M + m2 + 1)
#        for c in range(C):
#            class_mask = (y==c)
#            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
#            if m1==M-1:
#                xlabel(attributeNames[m2])
#            else:
#                xticks([])
#            if m2==0:
#                ylabel(attributeNames[m1])
#            else:
#                yticks([])
#                
#            #ylim(0,X.max()*1.1)
#            #xlim(0,X.max()*1.1)
#legend(classNames)


# Plotting only for specific selectred attributes
## Next we plot a number of atttributes
Attributes = [0,2,3,5,7]
NumAtr = len(Attributes)

figure(figsize=(13,13))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
            if m1==NumAtr-1:
                xlabel(attributeNames[Attributes[m2]],fontsize=14)
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[Attributes[m1]],fontsize=14)
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)
plt.savefig('ole.png')


#show()


#plot(np.array(X[0,2]), np.array(X[class_mask,m1]), '.')
#show()

print('Ran Exercise 4.1.5')