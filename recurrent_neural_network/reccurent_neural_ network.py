# Recurrent Neural Network

# Part 1 : Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing th training set
dataset_train = pd.read_csv('dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 time steps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 : BUilding the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding the first LSTN and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape =(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the second LSTN and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))


# Adding the third LSTN and some Dropout regularisation


# Adding the fourth LSTN and some Dropout regularisation



#Part 3 :  MAking the predictions and visualizing the results
