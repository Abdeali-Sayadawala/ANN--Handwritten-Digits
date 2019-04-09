# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import np_utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# importing and preprocessing the mnist dataset for digit recognition
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# trainig and testing image data preprocessing
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# training and testing label data preprocessing
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#----------------------Artificial Neural Network--------------#
# Making the Network
Network = Sequential()
# 1st Hidden Layer
Network.add(Dense(512, input_shape=(784,), activation='relu'))
Network.add(Dropout(0.2))
# 2nd Hidden Layer
Network.add(Dense(512, activation='relu'))
Network.add(Dropout(0.2))
# 3rd Hidden Layer
Network.add(Dense(512, activation='relu'))
Network.add(Dropout(0.2))
# 4th Hidden Layer
Network.add(Dense(512, activation='relu'))
Network.add(Dropout(0.2))
# Output Layer
Network.add(Dense(10, activation='softmax'))
#----------------------Artificial Neural Network--------------#

# Compiling the Network
Network.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Training
Network.fit(x_train, y_train, batch_size=128, nb_epoch=30)

# saving the model
Network.save('my_model.h5')
