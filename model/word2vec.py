from curses import window
from gensim.models import Word2Vec as w2v
from model.mean_embeding_vectorizer import MeanEmbeddingVectorizer

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
