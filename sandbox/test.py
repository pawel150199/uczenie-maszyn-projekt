import pandas as pd
from helpers.text_preprocessing import TextPreprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from keras.utils.vis_utils import  plot_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf




if __name__ == "__main__":
    df = pd.read_csv("data/email.csv")

    # Create list of sentences
    sentences = list(df["Message"])
    y = list(df["Category"])

    preprocessing = TextPreprocessing()
    preprocessed_data = preprocessing.preprocess(sentences)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_data)
    print(X)
    
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.6, random_state=1410)
    print(f"\nTest set size: {X_test.shape}\nTrain set size: {X_train.shape}\n")

    print(X.shape[1])

    model = Sequential()
    model.add(Dense(10, input_dim=X.shape[1], activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    plot_model(model, show_shapes=True, show_layer_names=True)
    
    history = model.fit(X_train.toarray(), np.array(y_train), epochs=30, batch_size=10, verbose=False)


    loss, accuracy = model.evaluate(X_train.toarray(), np.array(y_train), verbose=False)
    print(f"Train Accuracy: {accuracy}")

    loss, accuracy = model.evaluate(X_test.toarray(), np.array(y_test), verbose=True)
    print(f"Test Accuracy: {accuracy}")

    cl = KNeighborsClassifier()
    cl.fit(X_train, y_train)

    y_pred = cl.predict(X_test)

    acc = balanced_accuracy_score(y_test, y_test)

    print(f"Original: {y_test}")
    print(f"Predicted: {y_pred}")

    print(f"Accuracy score: {acc}")




    