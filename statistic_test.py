from distutils.log import error
import numpy as np
import warnings
import os
from tabulate import tabulate
from scipy.stats import ttest_ind
from scipy.stats import rankdata, ranksums


def main(table_name: str):
    warnings.filterwarnings("ignore")

    alpha=0.05 
    m_fmt="%3f" 
    std_fmt=None 
    nc="---" 
    db_fmt="%s"
    tablefmt="plain"

       
    scores = np.load("results/main-scores.npy")
    mean_scores = np.load("results/main-mean.npy")
    std =np.load("results/main-std.npy")
    metrics = ["accuracy"]
    vecs = ["TFIDF", "BoW", "HV", "TFIDF-B"]
    n_clfs = len(vecs)
    t = []


    # VECTORIZER X CLASSIFICATOR X METRIC
    t.append(["%.3f" % v for v in mean_scores])

    # If std_fmt is not None, std will appear in tables
    if std_fmt:
        t.append([std_fmt % v for v in std])
    
    T, p = np.array(
        [[ttest_ind(scores[i, :, :],
            scores[j, :, :])
        for i in range(len(vecs))]
        for j in range(len(vecs))]
    ).swapaxes(0, 2)
    _ = np.where((p < alpha) * (T > 0))
    conclusions = [list(1 + _[1][_[0] == i])
                for i in range(n_clfs)]
                
    t.append([", ".join(["%i" % i for i in c])
                    if len(c) > 0 and len(c) < len(vecs)-1 else ("all" if len(c) == len(vecs)-1 else nc)
                    for c in conclusions])

    # Show outputs  
    headers = ['Metric']
    for i in vecs:
        headers.append(i)

    with open('%s.txt' % (table_name), 'w') as f:
            f.write(tabulate(t, headers, tablefmt='latex'))

    # Show outputs
    print(tabulate(t, headers=vecs, tablefmt='grid'))

if __name__ == "__main__":
    main("tables/main")