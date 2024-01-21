import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import gutenberg
from multiprocessing import Pool
from scipy import spatial
from sklearn.base import ClassifierMixin, BaseEstimator

class TextToImageClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, X, y):
        pass
    
    def _word2vec(self, *args):
        pass

    def _padding(self, *args):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass