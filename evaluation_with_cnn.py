import ast
from tabnanny import verbose
import pandas as pd 
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from model.doc2vec import Doc2Vec
from model.word2vec import Word2Vec
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
import multiprocessing
from model.nn import get_mlp
from model.cnn import get_model
from text_to_image import TextToImage

from model.mean_embeding_vectorizer import MeanEmbeddingVectorizer

RANDOM_STATE = 1410
VECTOR_SIZE = 40
EPOCHS = 40
CORES = multiprocessing.cpu_count()


def main():
    # Ignore warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv("data/preprocessed_imdb.csv")

    vecs = {
        "TFIDF" : TfidfVectorizer(),
        "BoW" : CountVectorizer(binary=True),
        "HV" : HashingVectorizer(n_features=800, binary=True),
        "BTFIDF" : TfidfVectorizer(binary=True),
        "D2V" : Doc2Vec(vector_size=VECTOR_SIZE, min_count=1),
        "W2V" : Word2Vec(vector_size=VECTOR_SIZE, min_count=1, workers=CORES-1),
        "TTI" : TextToImage(vector_size=VECTOR_SIZE, epochs=EPOCHS)
    }

    metrics = {
        'AC' : accuracy_score,
    }

    n_splits = 2 
    n_repeats = 2
    scores = np.zeros((len(vecs), n_splits * n_repeats, 1, len(metrics)))


    rskf = rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    print("\nExperiment evaluation\n")

    for vec_id, vec_name in enumerate(vecs):
        X = np.array(df["lematized_tokens"])[:100]
        y = np.array(df["sentiment"])[:100]
    
        print(vec_name)
        vectorizer = vecs[vec_name]
        X = vectorizer.fit_transform(X)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            if vec_name == "TTI":
                clf = get_model(X[train].shape[1], X[train].shape[1])
                clf.fit(X[train], y[train], epochs=EPOCHS, verbose=0)
                y_pred = clf.predict(X[test], verbose=0).round()
                y_pred = np.argmax(np.array(y_pred), axis=1)
                for metric_id, metric_name in enumerate(metrics): 
                    # VECTORIZER X FOLD X CLASSIFICATOR X METRIC
                    scores[vec_id, fold_id, 0, metric_id] = metrics[metric_name](np.array(y[test]),np.array(y_pred))
            else:
                clf = MLPClassifier(hidden_layer_sizes=50)

                print(X[train].shape)
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                for metric_id, metric_name in enumerate(metrics): 
                    # VECTORIZER X FOLD X CLASSIFICATOR X METRIC
                    scores[vec_id, fold_id, 0, metric_id] = metrics[metric_name](np.array(y[test]),np.array(y_pred))


    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)
    
    print("\nSave results...\n")
    np.save(f"results/nn-scores", scores)
    np.save(f"results/nn-mean", mean)
    np.save(f"results/nn-std", std)
    print("\nResults saved\n")

    print(scores)

if __name__ == "__main__":
    main()

