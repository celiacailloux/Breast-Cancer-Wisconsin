#------------------------------------------- Loading data for linear regression
# Choose the desired attribute for which regression should be performed on

# exercise 5.2.4
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm

# requires wine data from exercise 5.1.5
from LoadingData import *

# ------------------------------- Split dataset into features and target vector
# att_reg_idx is the desired attribute that regression will be performed on
att_reg_idx = attributeNames.index('Mean Area')
y = X[:,att_reg_idx]

X_cols = list(range(0,att_reg_idx)) + list(range(att_reg_idx+1,len(attributeNames)))
X = X[:,X_cols]
attributeNames = attributeNames[:att_reg_idx] + attributeNames[att_reg_idx+1 :]

N = len(y)
M = len(attributeNames)
C = len(classNames)
# -------------------------------  Fit ordinary LINEAR least squares regression model
'''
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Alcohol content (true)'); ylabel('Alcohol content (estimated)');
subplot(2,1,2)
hist(residual,40)

show()
'''

print('Ran Exercise LoadingDatRegAutomated.py')