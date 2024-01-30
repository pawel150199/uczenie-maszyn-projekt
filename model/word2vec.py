from curses import window
from os import truncate
import numpy as np
from gensim.models import Word2Vec as w2v
from model.mean_embeding_vectorizer import MeanEmbeddingVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Word2Vec(object):
    def __init__(self, vector_size=30, min_count=2, epochs=50, window=5, workers=1) -> None:
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.window = window
        self.workers = workers

    def fit_transform(self, X):
        X = [x.split() for x in X]

        model = w2v(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        model.build_vocab(X)
        model.train(X, total_examples=model.corpus_count, epochs=self.epochs)

        mev = MeanEmbeddingVectorizer(model)
        X_t = mev.transform(X)

        return X_t
    
    def fit_transform_and_return_model(self, X):
        X = [x.split() for x in X]

        model = w2v(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        model.build_vocab(X)
        model.train(X, total_examples=model.corpus_count, epochs=self.epochs)

        mev = MeanEmbeddingVectorizer(model)
        X_t = mev.transform(X)
        
        return X_t, model
    
    def fit(self, X, max_length=None):
        # The same as fit_transform but not meaning values
        X = [x.split() for x in X]

        model = w2v(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        model.build_vocab(X)
        model.train(X, total_examples=model.corpus_count, epochs=self.epochs)
        word2vec = {word: model.wv[word] for word in model.wv.index_to_key}

        vectors = []

        for words in X:
            word_vectors = [word2vec[w] for w in words]
            vector = np.array(word_vectors)
            vectors.append(vector)

        vectors = pad_sequences(vectors, padding='post', truncating='post', dtype='float32', maxlen=max_length)

        #vectors = pad_sequences(np.array([
        #    np.array([word2vec[w] for w in words]
        #            or [np.zeros(dim)])
        #    for words in X
        #]), 
        #    padding='post',
        #    dtype='float32',
        #    maxlen=max_length
        #)

        print(np.array(vectors).shape)
        
        return np.array(vectors)
