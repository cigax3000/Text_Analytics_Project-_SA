from src.features.process_text.lemmatization import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def compute_tfidf(dataset):
    tfidf = TfidfTransformer()
    idfs = tfidf.fit_transform(dataset)

    return tfidf, idfs


def compute_tfidf_stopwords(dataset, stopwords_lang):
    tfidf = TfidfTransformer(stop_words=stopwords_lang)
    idfs = tfidf.fit_transform(dataset)

    return tfidf, idfs


def compute_tfidfle_stopwords(dataset, stopwords_lang):
    tfidf = TfidfTransformer(tokenizer=tokenize, stop_words=stopwords_lang)  # e.g. stop_words = 'english'
    idfs = tfidf.fit_transform(dataset)

    return tfidf, idfs

