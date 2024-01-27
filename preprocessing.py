from cProfile import label
from turtle import write_docstringdict
import pandas as pd
import string
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import  Counter
import logging
import warnings


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
#nltk.download('stopwords')
#nltk.download('wordnet')
import warnings
warnings.filterwarnings('ignore')

def remove_html_tags(data):
    pattern= re.compile('<.*?>')
    return pattern.sub(r'', data)

def remove_urls(data):
    pattern= re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', data)

def remove_punctuations(data):
    exclude = string.punctuation
    return data.translate(str.maketrans('', '', exclude))

def remove_between_square_brackets(data):
    return re.sub('\[[^]]*\]', '', data)

def remove_stopwords(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))

    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)

def stem_words(data):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in data])

def lemmit(data):
    wordnet_lem = WordNetLemmatizer()
    return " ".join([wordnet_lem.lemmatize(word) for word in data])

def denoise_text(data):
    text = remove_html_tags(data)
    text = remove_urls(text)
    text = remove_punctuations(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text

def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def main():
    path = "data/imdb.csv"
    df = pd.read_csv(path)

    # Checking for any missing values
    print(f"\nMissing values: {df.isna().sum()}")

    # Some plots before cleaning data

    # Count of good and bad reviews
    count=df['sentiment'].value_counts()
    print('Total Counts of both sets'.format(),count)

    # Denoise text  
    df["review"] = df["review"].apply(denoise_text)

    # Tokenize
    df['tokens'] = df['review'].str.lower().apply(word_tokenize)

    # Lematize
    df['lematized_tokens'] = df['tokens'].apply(lambda x: lemmit(x))

    # Stemmed
    df['stemmed_tokens'] = df['tokens'].apply(lambda x: stem_words(x))
    
    df.sentiment.replace("positive" , 1 , inplace = True)
    df.sentiment.replace("negative" , 0 , inplace = True)

    print(df.head())

    # Save to file
    df.to_csv("data/preprocessed_imdb.csv")

if __name__ == "__main__":
    main()