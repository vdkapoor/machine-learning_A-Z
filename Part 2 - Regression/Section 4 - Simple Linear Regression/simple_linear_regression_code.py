# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 01:13:53 2017

@author: Varun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

path= 'C:\Users\Varun\Desktop\Varun\ML\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Salary_Data.csv'
dataset=pd.read_csv(path)


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
              
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5, random_state = 0)

#fitting regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predict the values
Y_pred=regressor.predict(X_test)

regressor.coef_
regressor.intercept_
#plots
plt.scatter(X_train,Y_train,color='r')
plt.plot(X_test,Y_pred,color='blue')
plt.title('Salary vs Experience(Training Test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show









  
