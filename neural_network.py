from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf


def create_cnn(input_dim: int):
    input_dim = input_dim
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model