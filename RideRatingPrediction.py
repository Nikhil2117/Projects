# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

pd.set_option('display.max_columns',100)
train_df = pd.read_csv('E:\pyWork\pyProjects\sippython\Project\FinalProjects\Price_prediction_model\\train.csv',sep=',')
test_df = pd.read_csv('E:\pyWork\pyProjects\sippython\Project\FinalProjects\Price_prediction_model\\test.csv',sep=',')

train_df.head(2)
train_df.columns
train_df.shape
#Now Finding the Average speed in the data as it suppose to be useful in prediction
#also we know that average speed = total distance travelled/Total Time taken
#therefore creating new column Average Speed
train_df['Average_Speed'] = train_df['distance_travelled']/(train_df['duration_time'])
train_df['Average_Speed'].isna().sum()
train_df = train_df[~train_df['Average_Speed'].isna()]
train_df['Average_Speed'].isna().sum()
train_df.Average_Speed
train_df.isna().sum()
train_df.shape
train_df = train_df.replace([np.inf, -np.inf], np.nan)
train_df = train_df[~train_df['Average_Speed'].isna()]
#%%
#Prediction model for customer if he is gonna rate or not
train_df.dtypes
train_df.info()
train_df.columns
train_df.head()
#now selecting independent variables for which out dependent column (rated) depends
x_train = train_df[['customer_id','driver_id','booking_source','car_type','distance_travelled','duration_time','Average_Speed']]
y_train = train_df[['was_rated','rating']]
x_train.isna().sum()
y_train.isna().sum()

from sklearn.model_selection import train_test_split
a_train,a_test,b_train,b_test = train_test_split(x_train, y_train, test_size=0.3)

from sklearn.linear_model import LogisticRegression
rated_model = LogisticRegression(random_state=0)
rated_model.fit(a_train,b_train.iloc[:,0])
y_pred_train = rated_model.predict(a_train)
rated_model.score(a_train,b_train.iloc[:,0])

rated_model.score(a_test,b_test.iloc[:,0])


#now selecting variables using statsmodel
import statsmodels.api as sm
model = sm.Logit(y_train.iloc[:,0],x_train)
results = model.fit()
results.summary()

x_train1 = train_df[['customer_id','driver_id','booking_source','distance_travelled','duration_time','Average_Speed','car_type']]
y_train1 = train_df[['was_rated','rating']]
x_train1.isna().sum()
y_train1.isna().sum()

from sklearn.model_selection import train_test_split
a_train1,a_test1,b_train1,b_test1 = train_test_split(x_train1, y_train1, test_size=0.3)


from sklearn.ensemble import RandomForestClassifier
rated_model1 = RandomForestClassifier(random_state=0,n_estimators=2,max_features=7,max_depth=10)
rated_model1.fit(a_train1,b_train1.iloc[:,0])
y_pred_train1 = rated_model.predict(a_train1)
rated_model1.score(a_train1,b_train1.iloc[:,0])

rated_model1.score(a_test1,b_test1.iloc[:,0])
#Since the random_forest giving accuracy score in test set same as train test this means 
#it is predicting the results better.
#therefore going with random_forest_classifier  model


#Now the LogisticModel is Ready Now Implementing It on Test Data
#First Creating Average_speed columns in Test dataset
test_df['Average_Speed'] = test_df['distance_travelled']/(test_df['duration_time'])
test_df['Average_Speed'].isna().sum()
test_df = test_df[~test_df['Average_Speed'].isna()]
test_df['Average_Speed'].isna().sum()
test_df.Average_Speed
test_df.isna().sum()
test_df.shape
test_df = test_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df[~test_df['Average_Speed'].isna()]
test_df.shape
x_test = test_df[['customer_id','driver_id','car_type','booking_source','distance_travelled','duration_time','Average_Speed']]
y_pred_test = rated_model1.predict(x_test)
y_pred_test
#Now Inserting it into the test dataframe

test_df['was_rated'] = y_pred_test
test_df
#%%
#Now making models for Ratings of rides
x_train2 = train_df[['customer_id','driver_id','booking_source','car_type','distance_travelled','duration_time','Average_Speed']]
y_train2 = train_df[['was_rated','rating']]
x_train2.isna().sum()
y_train2.isna().sum()

from sklearn.model_selection import train_test_split
a_train2,a_test2,b_train2,b_test2 = train_test_split(x_train2, y_train2, test_size=0.3)

from sklearn.linear_model import LogisticRegression
rating_model = LogisticRegression(random_state=0)
rating_model.fit(a_train2,b_train2.iloc[:,1])
y_pred_train2 = rating_model.predict(a_train2)
rating_model.score(a_train2,b_train2.iloc[:,1])

rating_model.score(a_test2,b_test2.iloc[:,1])

#since the accuracy is not good enough therefore trying for the radom forest classifier

from sklearn.ensemble import RandomForestClassifier
rating_model1 = RandomForestClassifier(n_estimators=10 ,max_features=7,max_depth=10)
rating_model1.fit(a_train2,b_train2.iloc[:,1])
y_pred_train2 = rating_model1.predict(a_train2)
rating_model1.score(a_train2,b_train2.iloc[:,1])

rating_model1.score(a_test2,b_test2.iloc[:,1])
test_df['Average_Speed'] = test_df['distance_travelled']/(test_df['duration_time'])
test_df['Average_Speed'].isna().sum()
test_df = test_df[~test_df['Average_Speed'].isna()]
test_df['Average_Speed'].isna().sum()
test_df.Average_Speed
test_df.isna().sum()
test_df.shape
test_df = test_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df[~test_df['Average_Speed'].isna()]
test_df.shape
x_test = test_df[['customer_id','driver_id','car_type','booking_source','distance_travelled','duration_time','Average_Speed']]
y_pred_test3 = rating_model1.predict(x_test)
y_pred_test3
#Now Inserting it into the test dataframe

test_df['rating'] = y_pred_test3
test_df
