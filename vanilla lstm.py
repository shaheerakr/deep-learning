#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 03:37:09 2019

@author: shaheer
"""

from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
import numpy as np

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

#input sequence
raw_seq = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
# choose a number of time steps
n_steps = 3

#split into training and testing set
X,y = split_sequence(raw_seq,n_steps)

#sumarizing the data
for i in range(len(X)):
	print(X[i], y[i])

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

#define model
model = Sequential()

#adding the first hiddin layer
model.add(LSTM(64,activation='relu',input_shape = (n_steps,n_features)))

#adding dropout
model.add(Dropout(0.2))

#adding the output layer
model.add(Dense(1))

#compiling the model
model.compile(optimizer='adam',loss = 'mse')


#fiting the model
model.fit(X,y,epochs=200,verbose=True)

#unseen data
X_input = np.array([80,90,100])
X_input = X_input.reshape(1,n_steps,n_features)

y_pred = model.predict(X_input)








