from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

class NN(tf.Module):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=self.input_dim, activation="relu"))
        self.model.add(Dense(8, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def fit(self, X: np.array, y: np.array):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.array) -> np.array:
        y_pred = self.model.predict(X)
        return y_pred