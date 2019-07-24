# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(nikhil2117)s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('E:\pyWork\pyProjects\sippython\data\Airbnb_data.csv')

df.head()
df.shape
print(df.columns.tolist(),end=' ')
#checking null values:
df.isna().sum()

df[[df.dtypes == 'object']]
x = df[['number_of_reviews','bathrooms','latitude','longitude','bedrooms','beds']]
y= df[['log_price']]
x.isna().sum()
x['bathrooms'].fillna(0,inplace = True)
x['bedrooms'].fillna(0,inplace=True)
x['beds'].fillna(0,inplace=True)
y.fillna(0)
y.isna().sum()
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
#model is completed here

#train_test method
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

xtrain = pd.DataFrame(x_train)
ytrain = pd.DataFrame(y_train)
xtest = pd.DataFrame(x_test)
ytest = pd.DataFrame(y_test)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
regressor.coef_
y_test_pred= pd.DataFrame(regressor.predict(x_test),columns= ['y_test_pred'])

ytest

y_train_pred= pd.DataFrame(regressor.predict(x_train),columns= ['y_train_pred'])
