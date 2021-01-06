# Exercise 4.1.3

#from matplotlib.pyplot import boxplot, xticks, ylabel, title, show
import matplotlib.pyplot as plt
# requires data from exercise 4.1.1
from LoadingData import *#ex4_1_1 import *

plt.figure(figsize=(7,4))
# Convert X matrix to array
plt.boxplot(np.asarray(X[0,11]))

plt.xticks(range(0, 10),attributeNames, rotation = 45, fontsize=10)
plt.ylabel('cm',fontsize=14)
#plt.title(data_title + ' - boxplot')

#plt.show()
plt.savefig('boxplot.png')

print('Ran Exercise 4.1.3')