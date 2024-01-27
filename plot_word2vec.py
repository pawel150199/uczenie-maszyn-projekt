import pandas as pd
import numpy as np
import ast
from gensim.models import Word2Vec
import multiprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


CORES = multiprocessing.cpu_count()
VECTOR_SIZE = 32
ANNOTATIONS = False

def map_words_to_vectors(words, model, max_length=600):
    words = [x.split() for x in words]
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


X = np.array(df["lematized_tokens"])[:100]
y = np.array(df["sentiment"])[:100]


model = Word2Vec(vector_size=VECTOR_SIZE, window=5, min_count=1, workers=CORES-1)
model.build_vocab([x.split() for x in X], progress_per=10000)
model.train([x.split() for x in X], total_examples=model.corpus_count, epochs=100)

X = model.wv.vectors
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# Create a scatter plot of the projection

plt.scatter(result[:, 0], result[:, 1], edgecolors='black', linewidth=1)
plt.title("Word embedings distribution in 2D")
plt.xlabel("X")
plt.ylabel("Y")
words = np.asarray(model.wv.index_to_key)

if ANNOTATIONS:
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.savefig("images/word2vec_embedigs.png")

# Create a 3D scatter plot

pca = PCA(n_components=3)
result = pca.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(result[:, 0], result[:, 1], result[:, 2], edgecolors='black', linewidth=1)
ax.set_title("Word embedings distribution in 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.savefig("images/word2vec_embedings_3D.png")


# 10 most popular embedings
pca = PCA(n_components=2)
result = pca.fit_transform(X)
most_popular = ['movie', 'film', 'one', 'like', 'good', 'even', 'would', 'time', 'really', 'see']
words = np.asarray(model.wv.index_to_key)

plt.figure()

for i, word in enumerate(words):
    if word in most_popular:
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        plt.scatter(result[i, 0], result[i, 1], edgecolors='black', linewidth=1)


plt.title("Word embedings distribution in 2D")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.savefig("images/10_most_popular_word_embedings.png")

# Create a 3D scatter plot for 10 most popular words
most_popular = ['movie', 'film', 'one', 'like', 'good', 'even', 'would', 'time', 'really', 'see']

pca = PCA(n_components=3)
result = pca.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, word in enumerate(words):
    if word in most_popular:
        ax.scatter(result[i, 0], result[i, 1], result[i, 2], edgecolors='black', linewidth=1)
        ax.text(result[i, 0], result[i, 1], result[i, 2], word,
                color='black', fontsize=8, ha='right', va='bottom')

ax.set_title("Word embedings distribution in 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.savefig("images/10_most_popular_word_embedings_3D.png")