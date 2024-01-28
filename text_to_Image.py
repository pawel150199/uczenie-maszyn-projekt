
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
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

    def transform(self, X):
        vectors = self._word2vec(X)
        for image_id, image in enumerate(vectors):
            plt.imshow(image, cmap='Greys')     
            plt.axis("off")
            plt.savefig(f"data/images/{image_id}.png", bbox_inches='tight', pad_inches=0, transparent=True)
                
if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed_imdb.csv")
    X = np.array(df["lematized_tokens"])[:100]
    y = np.array(df["sentiment"])[:100]

    tti = TextToImage(max_length=100)
    vec = tti.transform(X)
    
