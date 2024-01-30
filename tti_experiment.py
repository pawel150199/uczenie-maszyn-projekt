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
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


from image_dataloader import imageLoader
from model.word2vec import Word2Vec
from text_to_image import TextToImage
from model.cnn import get_model
from model.neural_network import create_cnn

imdb_path = "data/preprocessed_imdb.csv"
df = pd.read_csv(imdb_path)

X = np.array(df["lematized_tokens"])[:2000]
y = np.array(df["sentiment"])[:2000]

tti = TextToImage(vector_size=100, min_count=1, max_length=400)
X = tti._word2vec(X)


print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

# PCA if needed
#pca = PCA()
#pca.fit(X_train, y_train)
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)

dim = X_test.shape[1]

model = get_model(height=X_train.shape[1], width=X_train.shape[2])
history = model.fit(X_train, to_categorical(y_train), epochs=20, validation_data=(X_test, to_categorical(y_test)))

# Plot some data

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
