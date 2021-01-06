# exercise 5.2.6
# -------------------------------- Logistic Regression (no cross validation)
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm

# requires data from exercise 5.1.4
#from ex5_1_5 import *
from LoadingData import *





# ---------- Fit logistic regression model
model = lm.logistic.LogisticRegression()
model = model.fit(X,y)

# ---------- Classify wine as White/Red (y=0/y=1) 
# and assess probabilities
y_est = model.predict(X)
y_est_white_prob = model.predict_proba(X)[:, 0] # Probability for being white wine
#y_est_red_prob = model.predict_proba(X)[:, 1]  # Probability for being red wine

# ---------- New Data object to test on
# Define a new data object (new type of wine), as in exercise 5.1.7
x = np.array([6.9, 1.09, .06, 2.1, .0061, 12, 31, .99, 3.5, .44, 12]).reshape(1,-1)
# Evaluate the probability of x being a white wine (class=0) 
x_class = model.predict_proba(x)[0,0]
#x_class_red = model.predict_proba(x)[0,1] # Evaluate the probability of x being a red wine (class=0) 

# ----------- Misclassification
# Evaluate classifier's misclassification rate over entire training data
# Misclassification is ??????
misclass_rate = sum(np.abs(y_est - y)) / float(len(y_est))

# Display classification results
print('\nProbability of given sample being a white wine: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
# np.nonzero returns the elements that are nonzero
class0_ids = np.nonzero(y==0)[0].tolist()
class1_ids = np.nonzero(y==1)[0].tolist()

plot(class0_ids, y_est_white_prob[class0_ids], '.y')
plot(class1_ids, y_est_white_prob[class1_ids], '.r')
xlabel('Data object (wine samples $x_i$)'); ylabel('Predicted prob. of class White');
legend([className[0],className[1]])#['White', 'Red'])
ylim(-0.01,1.5)

show()

print('Ran Exercise 5.2.6')

''' ------------ CONCLUSION
The classification is not optimal! On the figure we see that some of the wine 
that are very likely to be white are in fact red.
'''