import os
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import layers

from image_dataloader import imageLoader
from text_to_image import TextToImage

imdb_path = "data/preprocessed_imdb.csv"
df = pd.read_csv(imdb_path)

X = np.array(df["lematized_tokens"])
y = np.array(df["sentiment"])

tti = TextToImage(max_length=300, vector_size=300)
X = tti._word2vec(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

model = Sequential([
    layers.Conv2D(filters=32, kernel_size=(5,5), padding="same", activation="sigmoid", input_shape=(100,100,1)),
    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="sigmoid"),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="sigmoid"),
    layers.Dense(2)])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(X_train, to_categorical(y_train), epochs=100, validation_data=(X_test, to_categorical(y_test)))

# Plot sove data

import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    x = range(1, len(acc) + 1) 
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, acc, label="Training")
    plt.plot(x, val_acc, label="Validation")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
  
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, label="Training")
    plt.plot(x, val_loss, label="Validation")
    plt.title("Loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("images/learning_history.png")

plot_history(history)
exit()
cls = KNeighborsClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
