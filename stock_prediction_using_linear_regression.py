#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Import necessary Libraries required by the algorithm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


#import dataset used for the algorithm
Tesla_dataset = pd.read_csv('TSLA.csv')


# # Analyzing Data - Tesla_dataset
# 

# In[24]:


#Declare variables x and y
x = Tesla_dataset[['High','Low', 'Open', 'Volume']].values
y = Tesla_dataset[['Close']].values


# In[25]:


#split data into training and testing subsets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# # Train Data - Tesla_dataset

# In[37]:


#create a model to train data and do prediction
tesla_model = LinearRegression()


# In[38]:


#fit model to get logistic regression
tesla_model.fit(x_train, y_train)


# In[39]:


print(tesla_model.coef_)


# In[40]:


print(tesla_model.intercept_)


# In[41]:


predicted = tesla_model.predict(x_test)


# In[31]:


print(predicted)


# In[32]:


dframe = pd.DataFrame({'Actual' : y_test.flatten(), 'Predicted' : predicted.flatten()})


# In[33]:


#Display sample values
dframe.head(15)


# In[34]:


#Calculate values
print('Mean Absolute Error :', metrics.mean_absolute_error(y_test, predicted))
print('Mean Squared Error :', metrics.mean_squared_error(y_test, predicted))
print('Root Mean Squared Error :', math.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[35]:


graph = dframe.head(20)


# In[36]:


#Display Actual value with Predicetd value in Barchart 
graph.plot(kind ='bar')


# In[ ]:




