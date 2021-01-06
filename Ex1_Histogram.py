# Exercise 4.1.2
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure, subplot, hist, xlabel, ylim, show, title, yticks
import numpy as np
# requires data from exercise 4.1.1
from LoadingData import *
from collections import Counter
from scipy import stats 
from scipy.optimize import curve_fit
from scipy.stats import norm


#The following lines plot histogram of all attributes.
'''
figure(figsize=(16,29))

u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(10):
    subplot(8,5,i+1)
    hist(X[:,i], color=(0.2, 0.9-i*0.02, 0.4))
    xlabel(attributeNames[i],fontsize=14)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    mu, std = norm.fit(X[:,i])
   # p = norm.pdf(x, mu, std)
   # plt.plot(x, p, 'k', linewidth=2)
    
    if i in range(10,20):
       #print(i)#Counter(X[:,i]) > N/2:
       #ylim(0, Counter(X[:,i])*1.1) 
       ylim(0,400)
    else:
       ylim(0,N/2)
plt.show() 

      
'''

#The following plots a gaussian peak.
plt.figure()    
  
hist(X[:0], color=(0.2, 0.9-1*0.02, 0.4))  
xlabel(attributeNames[0])
xmin, xmax = plt.xlim()
x = np.linspace(5, 30, 100)
mu, std = norm.fit(X[:,0])
p = norm.pdf(x, mu, std)
yhist, xhist = np.histogram(X[:,0], bins=np.arange(4096))

xh = np.where(yhist > 0)[0]
yh = yhist[xh]
def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))
popt,pcov = curve_fit(gaussian,xh, yh)
plt.plot(x, p, 'k', linewidth=2)
plt.show()  
'''        
xt = plt.xticks()[0]  
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(X[:,0]))

# lets try the normal distribution first
m, s = stats.norm.fit(X[:,0]) # get mean and standard deviation  
pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
plt.plot(lnspc, pdf_g, label="Norm") # plot it

plt.plot(yhist)
i = np.linspace(5, 30, 100)
plt.plot(i, gaussian(i, *popt))
plt.xlim(0, 300)

plt.show()
'''


# If you only want to show selected histograms of selected attributes. 
# m is a list containing the indices of the selected attributes
'''
figure(figsize=(14,9))
m = [1, 7, 10]
for i in range(len(m)):
    subplot(1,len(m),i+1)
    hist(X[:,m[i]],50)
    xlabel(attributeNames[m[i]])
    ylim(0, N/2) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i==0: title(data_title + ': Histogram (selected attributes)')

#show()
#plt.savefig('histogram.png')
'''
print('Ran Exercise 4.1.2')

                    #most_common_float = Counter(data[key]) #
                    #experimentData[key] = most_common_float.most_common()[0][0]
                    