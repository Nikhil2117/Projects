# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(nikhil2117)s
"""
import pandas as pd

pass_data = pd.read_csv('E:\pyWork\pyProjects\sippython\data\\titanic_project.csv')

pass_data['Cabin'].loc[~pass_data.Cabin.isnull()] = 1
pass_data['Cabin'].loc[pass_data.Cabin.isnull()] = 0
pass_data
pass_data['Sex'].replace('male',0,inplace=True)
pass_data['Sex']= pass_data['Sex'].replace('female',1)
mode_age = pass_data['Age'].mode()  
mode_age=int(mode_age)
pass_data['Age'].fillna(mode_age,inplace = True)
pass_data

x = pass_data[['Fare','Age','Sex','Pclass','Cabin']]
y = pass_data[['Survived','PassengerId']]
x.head()
y.head()
x = x.values
y = y.values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.469, random_state = 0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(x_train, y_train[:,0])
y_train[:,0]

accuracy = model.score(x_test,y_test[:,0])
accuracy

y_pred = model.predict(x_test) 
y_pred.shape

predicted_results = pd.DataFrame()

predicted_results['PassengerId'] = y_test['PassengerId'] 
predicted_results['Survived'] = y_pred

predicted_results.set_index('PassengerId',inplace= True)

predicted_results.to_csv('Titanic_pred.csv')
