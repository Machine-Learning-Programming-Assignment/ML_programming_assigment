# -*- coding: utf-8 -*-

#Description: This program is designed to predict the future price of Tesla 🚘

#Import the libraries import pandas as pd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

#Collect and clean the data
df = pd.read_csv('TSLA.csv')
df = df.dropna()

#See the data
df

#Show the data visually
df.plot(x="Date", y="Close")
plt.xticks(rotation=45)

#Create the model
model = RandomForestRegressor()

from operator import length_hint
#Tain the model
X = df[['Open','High','Low','Volume']]
X = X[:int(len(df)-1)]
y = df['Close']
y = y[:int(len(df)-1)]
model.fit(X,y)

RandomForestRegressor()

#Test the model
preditions = model.predict(X)
print('The model score is:', model.score(X,y))

#Make the predictions
new_data = df[['Open','High','Low','Volume']].tail(1)
prediction = model.predict(new_data)
print('The model predicts the last row or day to be:', prediction)
print('Actual valueis:', df[['Close']].tail(1).values[0][0])

# Define the predicted and actual values
predicted_value = prediction[0]
actual_value = df[['Close']].tail(1).values[0][0]

# Create a bar plot
plt.bar(['Predicted', 'Actual'], [predicted_value, actual_value])
plt.title('Predicted vs Actual Closing Stock Price')
plt.ylabel('Closing Price ($)')
plt.show()
