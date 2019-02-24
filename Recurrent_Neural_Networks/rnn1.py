    #Part 1: CLeaning and preprocessing data
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

#Creates a numpy array instead of a pandas series. 
# Use iloc instead of loc as it creates a 2d array instead of one which is what the fit_transform method of the MinMaxScalar class expects
training_set = dataset_train.iloc[:,1:2].values 
#print('Training set is of type:', type(training_set))
#print(training_set[:3])

#Normalize features
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler() 
training_set_scaled = sc.fit_transform(training_set)
#print(training_set_scaled)

#Create e data structure with ?60 timesteps and 1 output
x_train = []
y_train = []

#Create a for loop to use the 60 previous timesteps as training data to predict tomorrow's value
time_step = 60
for i in range(time_step,len(training_set)):
    x_train.append(training_set_scaled[(i-time_step):i, 0])
    y_train.append(training_set_scaled[i, 0])

#Make x_train and y_train numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
print(y_train)
print(len(y_train))