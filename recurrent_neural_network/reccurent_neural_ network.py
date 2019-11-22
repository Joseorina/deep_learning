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

# Part 2 : BUilding the RNN

#Part 3 :  MAking the predictions and visualizing the results