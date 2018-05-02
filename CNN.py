#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 20:19:42 2018

@author: root
"""

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Importing the Keras libraries and packages
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense #intialize random weights

# Intiallizing the CNN
classifier =Sequential()

# Adding Convolution lAYER
classifier.add(Convolution2D(32,(3,3),input_shape =(64,64,3),activation='relu'))# 32*3*3-> 32 feature detectors of 3*3 size

# Adding maxpooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding Flatten layer
classifier.add(Flatten())

#Adding Dense layers
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

#compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#processing Image data
from keras.preprocessing import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64, 64),batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',target_size=(64, 64),batch_size=32,class_mode='binary')

classifier.fit_generator(training_set,steps_per_epoch=8000,epochs=25,validation_data=test_set,validation_steps=2000)
