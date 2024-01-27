import pandas as pd
import numpy as np
import ast
from gensim.models import Word2Vec
import multiprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import gensim.downloader as api
from nltk.tokenize import word_tokenize


CORES = multiprocessing.cpu_count()
VECTOR_SIZE = 32
ANNOTATIONS = False

def map_words_to_vectors(words, model, max_length=600):
    word_vectors = []

    for i in words:
        for j in i:
            print(j)
        word_vectors.append([model.wv[x] for x in i])

    padded_vectors = pad_sequences(word_vectors, maxlen=max_length, padding='post', truncating='post', dtype='float32')

    # Calculate mean along the correct axis
    mean_vectors = np.mean(padded_vectors, axis=1)
    return mean_vectors


path = "data/preprocessed_imdb.csv"
df = pd.read_csv(path)


X = df["lematized_tokens"][:100].to_numpy(dtype=list)
y = df["sentiment"][:100].to_numpy()

print(X[0])
print(y)

wv = api.load('glove-twitter-50')

xd = sent_vec(X, wv)
print(xd.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xd, y, test_size=0.2,stratify=y)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
from sklearn import metrics
predicted = classifier.predict(X_test)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))