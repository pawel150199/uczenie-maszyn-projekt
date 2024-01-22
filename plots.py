import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from nltk.corpus import stopwords
from collections import  Counter
from wordcloud import WordCloud

from preprocessing import get_top_text_ngrams

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
            
    top = sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
    x,y = zip(*top)
    plt.figure(figsize=(10,10))
    plt.ylabel("Count", fontsize=15)
    plt.xlabel("Words", fontsize=15)
    plt.title("Most common stop words", fontsize=20)
    plt.bar(x,y, color='green')

def plot_top_non_stopwords_barchart(text):
    stop = set(stopwords.words('english'))
    
    new = text.str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]

    counter = Counter(corpus)
    most = counter.most_common()
    x, y = [], []
    for word,count in most[:10]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    plt.figure(figsize=(10,10))
    plt.ylabel("Count", fontsize=15)
    plt.xlabel("Words", fontsize=15)
    plt.title("Most common non stop words", fontsize=20)
    plt.bar(x, y, color='green')

def generate_word_cloud(text_data, name):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(f"images/{name}")


def main():
    path = "data/preprocessed_imdb.csv"
    df = pd.read_csv(path)
    df_bf = pd.read_csv("data/imdb.csv")

    sentiment_count = sns.countplot(x='sentiment', data=df)
    plt.savefig("images/sentiment_count.png")

    top_stopwords = plot_top_stopwords_barchart(df_bf['review'])
    plt.savefig("images/top_stopwords.png")

    top_nonstopwords = plot_top_non_stopwords_barchart(df['review'])
    plt.savefig("images/top_nonstopwords.png")

    df['word_length'] = df['tokens'].apply(len)
    most_common_bi = get_top_text_ngrams(df.review, 20, 2)
    most_common_bi = dict(most_common_bi)
    temp = pd.DataFrame(columns = ["Common_words" , 'Count'])
    temp["Common_words"] = list(most_common_bi.keys())
    temp["Count"] = list(most_common_bi.values())

    # Create a bar chart using Matplotlib
    fig = px.bar(temp, x="Count", y="Common_words", title='Commmon Bigrams in Text', orientation='h', width=700, height=700,color='Common_words')
    fig.write_image("images/common_bigrams_in_text.png")
        
    # Plotting the distribution of review word counts
    plt.figure(figsize=(10, 5))
    print(df.head())
    review_word_counts = df['review'].apply(lambda review: len(review.split()))
    print(review_word_counts)
    plt.hist(review_word_counts, bins=50, color='green', alpha=0.7)
    plt.title('Distribution of Review Word Counts in IMDB Reviews')
    plt.xlabel('Review Word Count')
    plt.ylabel('Number of Reviews')
    plt.savefig("images/words_count_distribution.png")


    # Plotting the distribution of review word counts positive vs negative
    df['word_count'] = df['review'].apply(lambda x: len(x.split()))
    positive_word_counts = df[df['sentiment'] == 1]['word_count']
    negative_word_counts = df[df['sentiment'] == 0]['word_count']
    plt.figure(figsize=(10, 5))
    plt.hist(positive_word_counts, bins=20, alpha=0.7, label='Positive Reviews', color='green')
    plt.hist(negative_word_counts, bins=20, alpha=0.7, label='Negative Reviews', color='red')
    plt.xlabel('Review Word Count')
    plt.ylabel('Number of Reviews')
    plt.title('Word Count Distribution of Positive vs. Negative Reviews')
    plt.legend()
    plt.savefig("images/words_count_distribution_positive_vs_negative.png")

    # WordCloud
    positive_reviews = df[df['sentiment'] == 1]['review']
    negative_reviews = df[df['sentiment'] == 0]['review']

    generate_word_cloud(positive_reviews, "wordcloud_positive")
    generate_word_cloud(negative_reviews, "wordcloud_negative")

if __name__ == "__main__":
    main()