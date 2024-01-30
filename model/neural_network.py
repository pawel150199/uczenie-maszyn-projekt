from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


def create_cnn(height: int, width: int):
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (2, 2), activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation="softmax"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model