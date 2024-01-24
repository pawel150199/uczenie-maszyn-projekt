import ast
from tabnanny import verbose
import pandas as pd 
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from gensim.models import Word2Vec
import multiprocessing

RANDOM_STATE = 1410
VECTOR_SIZE = 32
CORES = multiprocessing.cpu_count()

def map_words_to_vectors(words, model):
   vectors = []
   for i in words:
       vectors.append(np.mean([model.wv[str(x)] for x in i], axis=0))
   return np.array(vectors)


def main():
    # Ignore warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv("data/preprocessed_imdb.csv")

    # Create list of sentences
    X = np.array(df["review"])[:100]
    y = np.array(df["sentiment"])[:100]

    clfs = {
        'KNN' : KNeighborsClassifier(),
    }

    vecs = {
        "TFIDF" : TfidfVectorizer(),
        "BoW" : CountVectorizer(),
        "HV" : HashingVectorizer(),
        "BTFIDF" : TfidfVectorizer(binary=True),
        "W2V" : Word2Vec(vector_size=VECTOR_SIZE, window=5, min_count=1, workers=CORES-1),
    }

    #model = Word2Vec(vector_size=VECTOR_SIZE, window=5, min_count=1, workers=CORES-1)
    #model.build_vocab(X, progress_per=10000)
    #model.train(X, total_examples=model.corpus_count, epochs=10)
    #print(model.wv["movie"])
    #exit("Bye Bye")

    metrics = {
        'BAC' : accuracy_score,
    }

    n_splits = 2
    n_repeats = 2

    scores = np.zeros((len(vecs), n_splits * n_repeats, len(clfs), len(metrics)))


    rskf = rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    print("\nExperiment evaluation\n")

    for vec_id, vec_name in enumerate(vecs):
        X = np.array(df["tokens"])[:100]
        y = np.array(df["sentiment"])[:100]
        print(X.shape)
        print(X[0])

        if vec_name == "W2V":
            print(vec_name)
            vectorizer = vecs[vec_name]
            vectorizer.build_vocab(X, progress_per=10000)
            vectorizer.train(X, total_examples=vectorizer.corpus_count, epochs=10)
            X = map_words_to_vectors(X, vectorizer)
        else:
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

