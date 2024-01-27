import os
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def imageLoader(path: str) -> np.array:
    images = []
    length = len(os.listdir(path))
    df = pd.read_csv("data/preprocessed_imdb.csv")
    labels = df["sentiment"][:length]
    for i in range(length):
        image = np.array(Image.open(f"{path}/{i}.png"))
        images.append(np.array(Image.open(f"{path}/{i}.png")))
    return np.array(images), np.array(labels)
        


if __name__ == "__main__":
    path = "data/images"
    images, labels = imageLoader(path)    
    fig = plt.figure()
    fig.suptitle("A sample from the original dataset", fontsize=18)
    for i in range(16):
        idx = random.randint(0, 100)
        a = fig.add_subplot(4,4, i+1)
        plt.imshow(images[idx])
        a.set_title(labels[idx])
        a.axis('off')
    plt.tight_layout()
    plt.show()
