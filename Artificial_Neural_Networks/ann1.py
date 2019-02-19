#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:12:43 2019

@author: abdullah
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Explore data
#dataset.shape
#dataset.info()

#Step 1: Cleaning dataset
X = dataset.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1).values
y = dataset.loc[:,'Exited'].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Encode geography feature
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#Encode gender feature
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Convert geography column into dummy variable
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Remove one of the columns to avoid the dummy variable trap
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#step 2: Building ANN

#Import important libraries and modules from them
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #Function that randomly turns off neurons; prevent overfit

#Step 2a: Initializing an ANN named classifier
classifier = Sequential()

#Now lets add an input and a hidden layer to it
classifier.add(layer=Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11,)) #Units are avg of independent variables and output variable
#Add dropout layer to prevent overfitting 
classifier.add(layer=Dropout(rate=0.1))
#Add a second hidden layer
classifier.add(layer=Dense(units=6, kernel_initializer='uniform', activation='relu'))

#Add an output layer
classifier.add(layer=Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Step 2c: Fit ANN and train it on training data
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# #Step 3: Evaluating ANN performance

#Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print(y_pred)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Homework challenge 1: Predict whether a single customer will leave the bank?

#Predict likelihood of customer leaving a bank with the following characteristics:
# new_obs = np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]) #Make 1 entry a float to supress warning when standardizing

#Scale inputs as model was trained on scaled values.
# std_new_obs = sc.transform(new_obs) 
# y_new = classifier.predict(std_new_obs)
# print(y_new)

# #Part 3a: Using K fold CV to evaluate model performance
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score

# #Function to build ANN architecture created above
# def build_classifier():
#     #Initialize an ANN named classifier
#     classifier = Sequential()
#     #Now lets add an input and a hidden layer to it
#     classifier.add(layer=Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11,)) #Units are avg of independent variables and output variable
#     #Add a second hidden layer
#     classifier.add(layer=Dense(units=6, kernel_initializer='uniform', activation='relu'))
#     #Add an output layer
#     classifier.add(layer=Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#     #Compiling the ANN
#     classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     return classifier

# #Create estimator (object of class kerasclassifier) to predict exit status of customer
# classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)   
# accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1) 
# print('average accuracy: ',accuracies/len(accuracies))

#Part 4c: Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


#Function to build ANN architecture created above
def build_classifier():
    #Initialize an ANN named classifier
    classifier = Sequential()
    #Now lets add an input and a hidden layer to it
    classifier.add(layer=Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11,)) #Units are avg of independent variables and output variable
    #Add a second hidden layer
    classifier.add(layer=Dense(units=6, kernel_initializer='uniform', activation='relu'))
    #Add an output layer
    classifier.add(layer=Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    #Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


