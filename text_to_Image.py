
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from model.word2vec import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import multiprocessing

class TextToImage(object):
    def __init__(self, vector_size=100, random_state=1410, epochs=50, window=5, min_count=1, max_length=None):
        self.cores = multiprocessing.cpu_count()
        self.vector_size = vector_size
        self.random_state = random_state
        self.epochs = epochs
        self.window = window
        self.min_count = min_count
        self.max_length = max_length
    
    def _word2vec(self, X):
        model = Word2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.cores-1)
        vectors = model.fit(X, max_length=self.max_length)
        
        return vectors
    
    def fit_transform(self, X, value: int=40):
        vectors = self._word2vec(X)
        height = vectors.shape[1]
        width = vectors.shape[2]

        vectors = vectors.reshape(X.shape[0], height*width)
        pca = PCA(n_components=value*value)
        vectors = pca.fit_transform(vectors)
        vectors = vectors.reshape(vectors.shape[0], value, value)

        return np.array(vectors)
        
    def transform(self, X):
        vectors = self._word2vec(X)
        for image_id, image in enumerate(vectors):
            grayscale_image = image.squeeze()
            plt.imshow(grayscale_image, cmap='gray')     
            plt.axis("off")
            plt.savefig(f"data/images/{image_id}.png", bbox_inches='tight', pad_inches=0, transparent=True)
                
if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed_imdb.csv")
    X = np.array(df["lematized_tokens"])[:20]
    y = np.array(df["sentiment"])[:20]

    tti = TextToImage(vector_size=100, min_count=1)
    vec = tti.transform(X)
    
