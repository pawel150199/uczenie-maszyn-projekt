import pandas as pd
import string
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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
nltk.download('wordnet')
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
    return " ".join([ps.stem(word) for word in data.split()])

def lemmit(data):
    wordnet_lem = WordNetLemmatizer()
    words = word_tokenize(data)
    return " ".join([wordnet_lem.lemmatize(word) for word in words])

def denoise_text(data):
    text = remove_html_tags(data)
    text = remove_urls(text)
    text = remove_punctuations(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text

def plot_top_stopwords_barchart(text):
    stop = set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    from collections import defaultdict
    dic = defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1
            
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
    x,y=zip(*top)
    plt.figure(figsize=(10,10))
    plt.bar(x,y)

def plot_top_non_stopwords_barchart(text):
    stop = set(stopwords.words('english'))
    
    new = text.str.split()
    new = new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:50]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    plt.figure(figsize=(10,10))
    plt.bar(x, y)

def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def main(plot: bool = False):
    path = "data/imdb.csv"
    df = pd.read_csv(path)
    stop = stopwords.words('english')

    # Checking for any missing values
    print(f"\nMissing values: {df.isna().sum()}")

    # Some plots before cleaning data

    # Count of good and bad reviews
    count=df['sentiment'].value_counts()
    print('Total Counts of both sets'.format(),count)

    if plot:
        sentiment_count = sns.countplot(x='sentiment', data=df)
        plt.savefig("images/sentiment_count.png")

        top_stopwords = plot_top_stopwords_barchart(df['review'])
        plt.savefig("images/top_stopwords.png")

        top_nonstopwords = plot_top_non_stopwords_barchart(df['review'])
        plt.savefig("images/top_nonstopwords.png")

    df["review"] = df["review"].apply(denoise_text)
    df['tokenized'] = df['review'].apply(lambda x: word_tokenize(x))
    print(df.head())
    df['lematized_tokens']= df['tokenized'].apply(lemmit)
    print(df.head())
    df['final_tokens'] = df['lematized_tokens'].apply(lambda x: stem_words(x))
    
    df.sentiment.replace("positive" , 1 , inplace = True)
    df.sentiment.replace("negative" , 0 , inplace = True)
    print(df.head())
    df.to_csv("data/preprocessed_imdb.csv")
    
    if plot:
        # Features
        df['word_length'] = df['tokenized'].apply(len)
        most_common_bi = get_top_text_ngrams(df.review, 20, 2)
        most_common_bi = dict(most_common_bi)
        temp = pd.DataFrame(columns = ["Common_words" , 'Count'])
        temp["Common_words"] = list(most_common_bi.keys())
        temp["Count"] = list(most_common_bi.values())
        fig = px.bar(temp, x="Count", y="Common_words", title='Commmon Bigrams in Text', orientation='h', 
                        width=700, height=700,color='Common_words')
        fig.savefig("images/common_bigrams_in_text.png")

        word_len = sns.histplot(df[df['sentiment'] == "positive"]['word_length'], kde=False, bins=200)
        word_len = sns.histplot(df[df['sentiment'] == "negative"]['word_length'], kde=False, bins=200)

        plt.savefig("images/words_len_distribution.png")
    
    # Save preeprocessed data

if __name__ == "__main__":
    main(False)