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

from model.mean_embeding_vectorizer import MeanEmbeddingVectorizer

RANDOM_STATE = 1410
VECTOR_SIZE = 100
CORES = multiprocessing.cpu_count()


def main():
    # Ignore warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv("data/preprocessed_imdb.csv")

    clfs = {
        'KNN' : KNeighborsClassifier(),
        #'SVC' : SVC(random_state=RANDOM_STATE),
        #'MLP' : MLPClassifier(random_state=RANDOM_STATE),
    }

    vecs = {
        #"TFIDF" : TfidfVectorizer(),
        #"BoW" : CountVectorizer(),
        #"HV" : HashingVectorizer(),
        #"BTFIDF" : TfidfVectorizer(binary=True),
        #"D2V" : Doc2Vec(vector_size=VECTOR_SIZE, min_count=1),
        "W2V" : Word2Vec(vector_size=VECTOR_SIZE, min_count=1, workers=CORES-1),
    }

    metrics = {
        'AC' : accuracy_score,
    }

    n_splits = 5
    n_repeats = 2

    scores = np.zeros((len(vecs), n_splits * n_repeats, len(clfs), len(metrics)))


    rskf = rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    print("\nExperiment evaluation\n")

    for vec_id, vec_name in enumerate(vecs):
        X = np.array(df["lematized_tokens"])[:1100]
        y = np.array(df["sentiment"])[:1100]
    
        print(vec_name)
        vectorizer = vecs[vec_name]
        X = vectorizer.fit_transform(X)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                clf = clfs[clf_name]
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                for metric_id, metric_name in enumerate(metrics): 
                    # VECTORIZER X FOLD X CLASSIFICATOR X METRIC
                    scores[vec_id, fold_id, clf_id, metric_id] = metrics[metric_name](np.array(y[test]),np.array(y_pred))


    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)
    
    print("\nSave results...\n")
    #np.save(f"results/main-scores", scores)
    #np.save(f"results/main-mean", mean)
    #np.save(f"results/main-std", std)
    print("\nResults saved\n")

    print(scores)

if __name__ == "__main__":
    main()

