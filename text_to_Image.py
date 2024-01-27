import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import gutenberg
from multiprocessing import Pool
from scipy import spatial
from sklearn.base import ClassifierMixin, BaseEstimator
import matplotlib.pyplot as plt 
import ast
from tabnanny import verbose
import pandas as pd 
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from model.doc2vec import Doc2Vec
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import multiprocessing

class TextToImage(object):
    def __init__(self, vector_size=36, random_state=1410):
        self.cores = multiprocessing.cpu_count()
        self.vector_size = vector_size
        self.random_state = random_state
    
    def _map_words_to_vectors(self, words, model, max_length=100):
        word_vectors = []

        for i in words:
            word_vectors.append([model.wv[x] for x in i if x in model.wv])

        padded_vectors = pad_sequences(word_vectors, maxlen=max_length, padding='post', truncating='post', dtype='float32')
        print(np.array(padded_vectors).shape)
        return padded_vectors
    
    def _word2vec(self, X, window=5, min_count=2):
        X = [x.split() for x in X]
        model = Word2Vec(vector_size=self.vector_size, window=window, min_count=min_count, workers=self.cores-1)
        model.build_vocab(X)
        model.train(X, total_examples=model.corpus_count, epochs=40)
        mean_vectors = self._map_words_to_vectors(X, model)
        return mean_vectors

    def transform(self, X, y):
        vectors = self._word2vec(X)
        for image_id, image in enumerate(vectors):
            plt.imshow(image, cmap='Greys')
            plt.savefig(f"data/images/{image_id}.png")


if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed_imdb.csv")
    X = np.array(df["lematized_tokens"])[:1000]
    y = np.array(df["sentiment"])[:1000]

    tti = TextToImage()
    vec = tti.transform(X, y)
    
