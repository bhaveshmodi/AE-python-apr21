# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:35:53 2021

@author: Bhavesh
"""
import pandas as pd
import matplotlib.pyplot as plt

dict ={'Age':[18,19,20,21,22,23,24,25,26,27,28],'Height':[76.1,77,78.1,78.2,78.8,79.7,79.9,81.1,81.8,82.8,83.5]}

dataset=pd.DataFrame(dict,columns=['Age','Height'])
dataset
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Accuracy of the model

#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#Coefficient
regressor.coef_

# Intercept
regressor.intercept_

plt.scatter(dataset['Age'],dataset['Height'], marker='*')

plt.xlabel('age', fontsize=16)
plt.ylabel('height', fontsize=16)
plt.title('grouped scatter plot - height vs age',fontsize=20)


