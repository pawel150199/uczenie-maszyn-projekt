import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec as d2v, TaggedDocument
from nltk.tokenize import word_tokenize

class Doc2Vec(object):
    def __init__(self, vector_size=30, min_count=2, epochs=50, window=5) -> None:
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.window = window

    def fit_transform(self, X):
        X_tagged = [TaggedDocument(words=word_tokenize(doc.lower()),
                              tags=[str(i)]) for i,
               doc in enumerate(X)]
        model = d2v(vector_size=self.vector_size,
                        min_count=self.min_count, epochs=self.epochs, window=self.window)
        model.build_vocab(X_tagged)
        model.train(X_tagged,
                    total_examples=model.corpus_count,
                    epochs=model.epochs
        document_vectors = [model.dv[idx] for idx in range(len(X))] 
        
        return np.array(document_vectors)
