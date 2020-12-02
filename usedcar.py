# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

"""Pre-Processing"""

train_data = pd.read_csv('train-data.csv')
test_data = pd.read_csv('test-data.csv')

train_data.isnull().sum()

train_data = train_data[train_data['Engine'].notna()]
train_data = train_data[train_data['Power'].notna()]
train_data = train_data[train_data['Seats'].notna()]
train_data = train_data[train_data['Mileage'].notna()]

train_data = train_data.reset_index(drop=True)

test_data.isnull().sum()

test_data = test_data[test_data['Engine'].notna()]
test_data = test_data[test_data['Power'].notna()]
test_data = test_data[test_data['Seats'].notna()]

test_data = test_data.reset_index(drop=True)

X_train = train_data.iloc[:, 1:-2].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, 1:-1].values

X_train.shape

for i in range(0,5872):
  z = X_train[i][7].split()
  X_train[i][7] = float(z[0])

for i in range(0,5872):
  z = X_train[i][8].split()
  X_train[i][8] = float(z[0])
  w = X_train[i][9].split()
  X_train[i][9] = float(w[0])

print(X_train)

for i in range(0,5872):
  z = X_train[i][0].split()
  
  X_train[i][0] = z[0]

X_train[0,0]

for i in range(0,1201):
  z = X_test[i][7].split()
  X_test[i][7] = float(z[0])
  y = X_test[i][8].split()
  X_test[i][8] = float(y[0])
  w = X_test[i][9].split()
  X_test[i][9] = float(w[0])

print(X_test)

for i in range(0,1201):
  z = X_test[i][0].split()
  z[0] = z[0].lower()
  X_test[i][0] = z[0]

print(X_train)

"""Encoder & Dummy Variable"""

from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()
X_train[:,5] = labelenc.fit_transform(X_train[:,5])
X_train[:,0] = labelenc.fit_transform(X_train[:,0])

from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))
X_train = np.delete(X_train,[0],axis=1)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [13])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))
X_train = np.delete(X_train,[0],axis=1)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [17])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))
X_train = np.delete(X_train,[0],axis=1)

print(X_train[4,:])

X_train.shape

"""Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

scy = StandardScaler()
y_train = scy.fit_transform(y_train.reshape(-1,1))

print(X_train)
print(y_train)

"""TEST DATA SET"""

from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()
X_test[:,5] = labelenc.fit_transform(X_test[:,5])
X_test[:,0] = labelenc.fit_transform(X_test[:,0])

from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test))
X_test = np.delete(X_test,[0],axis=1)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [13])], remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test))
X_test = np.delete(X_test,[0],axis=1)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [17])], remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test))
X_test = np.delete(X_test,[0],axis=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
print(X_test)

"""Split Dataset"""

from sklearn.model_selection import train_test_split
X_train_split, X_test_split,y_train_split, y_test_split = train_test_split(X_train,y_train,test_size = 0.2,random_state = 0)

"""Modelsssssssssszzzzzzzzzzzz

Random Forest
"""

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 114 , random_state=0)
regressor.fit(X_train_split, y_train_split)

"""Accuracy"""

y_pred = regressor.predict(X_test_split)

y_pred = scy.inverse_transform(y_pred)
print(y_pred)

print("Accuracy on Traing set: ",regressor.score(X_train_split,y_train_split))
print("Accuracy on Testing set: ",regressor.score(X_test_split,y_test_split))

y_test_split = scy.inverse_transform(y_test_split)
print(y_test_split)

"""Support Vector"""

from sklearn.svm import SVR
regressor_two = SVR(kernel ='rbf')
regressor_two.fit(X_train_split,y_train_split)
y_pred_two = regressor_two.predict(X_test_split)

y_pred_two = scy.inverse_transform(y_pred_two)
print(y_pred_two)

print(scy.inverse_transform(y_test_split))
print("Accuracy on Traing set: ",regressor_two.score(X_train_split,y_train_split))
print("Accuracy on Testing set: ",regressor_two.score(X_test_split,y_test_split))

"""Predicting Test Dataset Prices"""

y_test_pred = regressor.predict(X_test)
y_test_pred = scy.inverse_transform(y_test_pred)
print(y_test_pred)
