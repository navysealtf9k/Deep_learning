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

#Reshape input so it conforms with input shape expected by keras(i.e. a 3D array whose 1st arg is batch_size, 2nd arg is time step and 3rd arg is the number of predictors to be fed into the RNN)
x_train = np.reshape(x_train, newshape=(x_train.shape[0],x_train.shape[1], 1 ))
# print(x_train.shape)
#print(type(x_train))


#Part 2: Building a RNN

#Impor keras libary to build recurrent network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialize RNN
regressor = Sequential()

#Add the first LSTM layer; Make sure it returns its output as we are building a stacked NN 
regressor.add(layer=LSTM(units = 50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2)) #Ignore a few outputs from the stacked LSTM layer by using dropout to regularize results


#Add a second LSTM layer whose input is output from the previous LSTM layer; its shape is specified by the units argument
regressor.add(layer=LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

#Add a third LSTM layer
regressor.add(layer=LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

#Add a fourth LSTM layer
regressor.add(layer=LSTM(units = 50))
regressor.add(Dropout(rate=0.2))

#Add an output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(x=x_train, y=y_train, epochs=100, batch_size=32)


#Part 3: Making and visualizing predictions of google's stock price

#Get test set data

# Import test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values #Creates a numpy array instead of a pandas series. 

#Get the predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

#Create input array containing 60 dayse before the first day of 2017 to train the model with
train_using_60_prior_obs = len(dataset_total) - len(dataset_test) - time_step
inputs = dataset_total[train_using_60_prior_obs:len(dataset_total)].values #Make it a numpy array
inputs = inputs.reshape(-1,1) #reshape so its the format expected

#Scale inputs; we call transform here as it uses the min and max values estimated from the training data to scale new inputs
inputs = sc.transform(inputs)

#Transform input values into shape required to be fed into predict 
x_test = []
#Use 60 previous timesteps as training data to predict google's stock price on each day in 2017
time_step = 60
max_step = time_step + len(dataset_test)
for i in range( time_step, max_step):
    x_test.append(inputs[(i-time_step):i, 0])
x_test = np.array(x_test) 
x_test = np.reshape(x_test, newshape=(x_test.shape[0],x_test.shape[1], 1 )) #Make it a 3D array

#Get predicted stock prices
predicted_stock_prices = regressor.predict(x_test)
predicted_stock_prices = sc.inverse_transform(predicted_stock_prices) #convert to original scale

#Visualize results
plt.plot(real_stock_price, color = 'Red', label = 'Real prices')
plt.plot(predicted_stock_prices, color = 'Blue', label = 'Predicted prices')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()