import nltk
from  nltk.tokenize import regexp_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

class TextPreprocessing(object):
    def __init__(self) -> None:
        nltk.download("stopwords")
        self.stemmer = SnowballStemmer("english")
        self.stopwords = stopwords.words("english")
        self.dataset = []

    def preprocess(self, dataset: list) -> list:
        """Preprocess text data using stemming and delete stopwords

        Args:
            dataset (list): text list which all sentences

        Returns:
            list: return list with preprocessed text data
        """   
        
        data = []
        for i in range(len(dataset)):
            tokens = regexp_tokenize(str(dataset[i]),r"\w+")
            stems = [self.stemmer.stem(token) for token in tokens]
            words_no_stopwords = [word for word  in stems if word not in self.stopwords]
            document = " ".join(words_no_stopwords)
            data.append(document)
        return data
