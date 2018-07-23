
# coding: utf-8

# In[3]:


# If we want to use the notebook github integration addon...
#https://github.com/sat28/githubcommit

# https://github.com/mortada/fredapi
# http://mortada.net/python-api-for-fred.html
from fredapi import Fred
fred = Fred(api_key='')
data = fred.get_series('SP500')


# In[5]:


# Get First release only (ignore revisions)
data = fred.get_series_first_release('GDP')
data.tail() #Returns the last set of data. Pandas command.


# In[6]:


latest_data = fred.get_series_latest_release('GDP') #essentially the same as above, but with a different call
data.tail()


# In[7]:


#Get Data as of date
fred.get_series_as_of_date('GDP', '6/1/2014')


# In[8]:


fred.search('potential gdp').T

