import pandas as pd 
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from strlearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold

from helpers.text_preprocessing import TextPreprocessing
from neural_network import NN

RANDOM_STATE = 1410

def main():
    # Ignore warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv("data/email.csv")

    # Create list of sentences
    sentences = list(df["Message"])
    y = list(df["Category"])

    clfs = {
        'KNN' : KNeighborsClassifier(),
    }

    vecs = {
        "TFIDF" : TfidfVectorizer(),
        "BoW" : CountVectorizer(),
        "HV" : HashingVectorizer(),
        "BTFIDF" : TfidfVectorizer(binary=True),
        "W2V" : 
    }

    metrics = {
        'BAC' : balanced_accuracy_score,
    }

    n_splits = 2
    n_repeats = 2

    scores = np.zeros((len(vecs), n_splits * n_repeats, len(clfs), len(metrics)))

    #preprocessing = TextPreprocessing()
    #preprocessed_data = preprocessing.preprocess(sentences)

    rskf = rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    for vec_id, vec_name in enumerate(vecs):
        vectorizer = vecs[vec_name]
        X = vectorizer.fit_transform(preprocessed_data)
        y = np.array(y)
        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                clf = clfs[clf_name]
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                for metric_id, metric_name in enumerate(metrics): 
                    # VECTORIZER X FOLD X CLASSIFICATOR X METRIC
                    scores[vec_id, fold_id, clf_id, metric_id] = metrics[metric_name](np.array(y[test]),np.array(y_pred))
    
    print(scores)

if __name__ == "__main__":
    main()

