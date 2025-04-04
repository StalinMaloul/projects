import nltk
import unicodedata
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import pos_tag, wordpunct_tokenize


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(unicodedata.category(char).startswith("P") for char in token)

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for (token, tag) in pos_tag(wordpunct_tokenize(document))
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        tags = {
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
            "J": wordnet.ADJ,
        }
        tag = tags.get(pos_tag[0], wordnet.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        return [" ".join(self.normalize(doc)) for doc in documents]
