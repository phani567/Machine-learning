# ANN
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 14:06:04 2018

@author: root
"""
# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_x_1 = LabelEncoder()
x[:,1]=labelEncoder_x_1.fit_transform(x[:,1])
labelEncoder_x_2 = LabelEncoder()
x[:,2]=labelEncoder_x_2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]

#splitting dataset into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense #intialize random weights

# Initialising the ANN
classifier =Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_shape=(11,)))

# Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='glorot_uniform',activation='relu'))

#Adding final layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid')) #use softmax if units are more

# compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #categorical_crossentropy if more outputs

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 10,epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

