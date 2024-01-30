import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import time
import os

def get_model(height: int, width: int):
    model = tf.keras.Sequential()#Sequential()
    model.add(Conv2D(64, kernel_size = (8, 3), activation='relu', input_shape=(height, width, 1)))
    model.add(BatchNormalization())
    
    #model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(BatchNormalization())
    
    
    #model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2)) ## 
    #
    #model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation = 'softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print('model prepared...')
    return model