#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:14:48 2019

@author: fahadtariq
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet = pd.read_csv('Position_Salaries.csv')

X = dataSet.iloc[: , 1:2].values
Y = dataSet.iloc[: , 2].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)


#linear_Regression

from sklearn.linear_model import LinearRegression
linreg1 = LinearRegression()
linreg1.fit(X,Y)

#polynomial linaer Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

linreg2 = LinearRegression()
 
linreg2.fit(X_poly,Y)

#Visualizing the Linear Regression eresults

plt.scatter(X,Y,color = 'red')
plt.plot(X,linreg1.predict(X), color = 'black')
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel('levels')
plt.ylabel('Salaries')
plt.show()

#Visualizing the Polynomial Regression
X_grid = np.arange(min(X),max(X) , 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,  color = 'red')
plt.plot(X_grid,linreg2.predict(poly_reg.fit_transform(X_grid)), color = 'black')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show

#predict the result with linear regression model

linreg1.predict(6.5)

#predict the result with polynomial regression model
linreg2.predict(poly_reg.fit_transform(6.5))


