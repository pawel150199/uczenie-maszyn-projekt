import numpy as np

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = {word: word2vec.wv[word] for word in word2vec.wv.index_to_key}
        self.dim = word2vec.vector_size

    def transform(self, X):
        mean_vectors = np.array([
            np.mean([self.word2vec[w] for w in words]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        return mean_vectors