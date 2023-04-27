#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#description: This program attempts to predict the future price of stock


# In[8]:


import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[9]:


#import dataset used for the algorithm
Tesla_dataset = pd.read_csv('TSLA.csv')


# In[12]:


#read in the data
df = pd.read_csv('TSLA.csv')
#set the date as the index
df =df.set_index(pd.DatetimeIndex(df['Date'].values))
#show the data
df


# In[34]:


future_days = 3


# In[35]:


#create a new column
df[str(future_days)+'_Day_Price_Forecast'] = df [['Close']].shift(-future_days)
#show the data
df[['Close', str(future_days)+'_Day_Price_Forecast']]


# In[36]:


x = np.array(df[['Close']])
x = x[:df.shape[0] - future_days]
print(x)


# In[37]:


y = np.array(df[str(future_days)+'_Day_Price_Forecast'])
y = y[:-future_days]
print(y)


# In[38]:


#Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[44]:


from sklearn.svm import SVR
svr_model = SVR(kernel='rbf', C=1e3, gamma =0.00001)
svr_model.fit(x_train, y_train)


# In[45]:


svr_model_confidence = svr_model.score(x_test, y_test)
print('svr_model accuracy:', svr_model_confidence)


# In[46]:


svm_prediction = svr_model.predict(x_test)
print(svm_prediction)


# In[47]:


print(y_test)


# In[49]:


plt.figure(figsize=(12,4))
plt.plot(svm_prediction, label='Prediction', lw=2, alpha=.7)
plt.plot(y_test, label='Actual', lw=2, alpha=.7)
plt.title('Prediction vs Actual')
plt.ylabel('Price in USD')
plt.xlabel('Time')
plt.legend()
plt.xticks(rotation=45)
plt.show


# In[ ]:





# In[ ]:




