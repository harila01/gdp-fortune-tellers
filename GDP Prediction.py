
# coding: utf-8

# In[3]:


# If we want to use the notebook github integration addon...
#https://github.com/sat28/githubcommit

# https://github.com/mortada/fredapi
# http://mortada.net/python-api-for-fred.html
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from decimal import *
import numpy as numpy
from pandas import DataFrame
figsize(20, 5)
fred = Fred(api_key='')


# In[4]:


# Quarterly GDP Growth plot
gdp = {}
gdp['Real GDP'] = fred.get_series('GDPC1')
gdp = pd.DataFrame(gdp)
gdp.plot()


# In[25]:


# Pulling of (potentially) relevant data
# List of dictionaries
economic_data = {}
economic_data.update(
    {'CPI' : fred.get_series('CPIAUCSL'),
     'Federal Funds' : fred.get_series('FEDFUNDS'),   
     'M2 Money Stock' : fred.get_series('M2NS'),
     'Unemployment Rate' : fred.get_series('UNRATE'),
     'Moody\'s Seasoned Aaa Corporate Bond Yield' : fred.get_series('AAA'), 
     'All Employees: Total Nonfarm Payrolls' : fred.get_series('PAYNSA'),
     'Industrial Production Index' : fred.get_series('INDPRO'),
     '1-Year Treasury Constant Maturity Rate' : fred.get_series('GS1'),
     'Personal Consumption Expenditures' : fred.get_series('PCE'),
     'Year Treasury Constant Maturity Rate' : fred.get_series('GS5'),
     'Personal Savings Rate' : fred.get_series('PSAVERT'),
     'Civilian Labor Force Participation Rate' : fred.get_series('LNU01300000'),
     'Interest Rates, Discount Rate for United States' : fred.get_series('INTDSRUSM193N'),
     'Federal Minimum Hourly Wage for Nonfarm Workers for the United States' : fred.get_series('FEDMINNFRWG')
    })
# Pulling of Y value as the last row.
economic_data['GDP Quarterly Growth'] = fred.get_series('A191RL1Q225SBEA')
economic_data = pd.DataFrame(economic_data)
#print(economic_data)
economic_data.plot()


# In[12]:


# Drop NaN values (debatable if we really want to do this long term)
economic_dropped = economic_data
economic_dropped.dropna(inplace=True)
print(economic_dropped)


# In[22]:


# Now we start working towards the model
# Assigning our X and Y variables
X = economic_dropped.values[:,0:-1]
Y = economic_dropped.values[:,-1]

# Normalize our X values. This does not seem to work well for Nonfarm Payrolls?
from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)
#print(normalized_X)
#print(Y)
plot_X = normalized_X
plot_X = pd.DataFrame(plot_X)
plot_X.plot()
# We normalize our Y values.


# In[33]:


# Model testing data set, validation sets creation
# Heavily pulled from:
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
from sklearn import model_selection
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(normalized_X, Y, test_size=validation_size, random_state=seed)
seed = 7
scoring = 'explained_variance' # This is subject for change.


# In[34]:


# Evaluating Different Algorithms
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

models = []
models.append(('LR', LinearRegression()))
models.append(('DTL', DecisionTreeRegressor()))

# Need to understand the loop a bit better . . . 
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[35]:


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


lnr = LinearRegression()
lnr.fit(X_train, Y_train)
Y_predictions = lnr.predict(X_validation)
print(lnr.intercept_)

from sklearn import metrics
print(metrics.mean_absolute_error(Y_predictions, Y_validation))
print(metrics.mean_squared_error(Y_predictions, Y_validation))
print(numpy.sqrt(metrics.mean_squared_error(Y_predictions, Y_validation)))

