import pandas as pd
import numpy as np
import ast
from gensim.models import Word2Vec
import multiprocessing
from tensorflow.keras.preprocessing.text import Tokenizer


CORES = multiprocessing.cpu_count()
VECTOR_SIZE = 32


        

path = "data/preprocessed_imdb.csv"
df = pd.read_csv(path)

X = df["tokens"][:100]
y = np.array(df["sentiment"])[:100]

X = ' '.join(X)


print(X[0])


def map_words(words, model):
    vectors = []
    for i in words:
        sentences = str(i).split()
        x = np.array([model.wv[x] for x in sentences])
        x = np.mean(x)
        vectors.append(x)
    return np.array(vectors)


model = Word2Vec(vector_size=VECTOR_SIZE, window=5, min_count=1, workers=CORES-1)
model.build_vocab(X, progress_per=10000)
model.train(X, total_examples=model.corpus_count, epochs=10)
X = map_words(X, model)
print(model.wv["movie"])
#print(X)
exit("Bye Bye")

print(X.shape)
print(y.shape)

print(X[0])
