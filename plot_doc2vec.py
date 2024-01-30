import pandas as pd
import numpy as np
from model.doc2vec import  Doc2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("data/preprocessed_imdb.csv")

X = np.array(df['lematized_tokens'])
y = np.array(df['sentiment'])


# train the Doc2vec model
model = Doc2Vec()
X_t = model.fit_transform(X)

print(X_t.shape)

pca = PCA(n_components=2)
result = pca.fit_transform(X_t)

plt.scatter(result[y == 0, 0], result[y == 0, 1], marker="o", color='blue', edgecolors='black', linewidth=0.2, label="negative")
plt.scatter(result[y == 1, 0], result[y == 1, 1], marker="o", color='red', edgecolors='black', linewidth=0.2, label="positive")
plt.title("Documents embedings distribution in 2D")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.savefig("images/doc2vec_embedings.png")


pca = PCA(n_components=3)
result = pca.fit_transform(X_t)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(result[y == 0, 0], result[y == 0, 1], result[y == 0, 2], marker="o", color='blue', edgecolors='black', linewidth=0.2, label="negative")
ax.scatter(result[y == 1, 0], result[y == 1, 1], result[y == 1, 2], marker="o", color='red', edgecolors='black', linewidth=0.2, label="positive")
ax.set_title("Documents embedings distribution in 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.tight_layout()
plt.savefig("images/doc2vec_embedings_3D.png")

