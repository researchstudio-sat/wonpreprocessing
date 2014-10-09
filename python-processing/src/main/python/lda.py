from os import listdir
from os.path import isfile, join

from gensim import matutils
from gensim.models import LdaModel
import numpy as np
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from feature_extraction import ScikitNltkTokenizerAdapter, default_pos_tagger
from mail_utils import mail_preprocessor


def dataset_newsgroups(categories=None):
    random = np.random.RandomState()
    data = fetch_20newsgroups(categories=categories, subset='train',
                              shuffle=True, random_state=random,
                              remove=('headers', 'footers', 'quotes'))
    tokenize = ScikitNltkTokenizerAdapter(lemmatizer=WordNetLemmatizer())
    return data.data, 'content', tokenize


def dataset_mails(path):
    filenames = [join(path, f) for f in listdir(path) if
                 isfile(join(path, f)) and f.endswith('.eml')]
    filenames = np.array(filenames)
    np.random.shuffle(filenames)
    tokenize = ScikitNltkTokenizerAdapter(preprocessor=mail_preprocessor,
                                          lemmatizer=WordNetLemmatizer(),
                                          pos_tagger=default_pos_tagger)
    return filenames, 'filename', tokenize


def fit_lda(X, vocabulary, n_topics=10, passes=1):
    """ Fit LDA from a scipy CSR matrix (X). """
    return LdaModel(matutils.Sparse2Corpus(X), num_topics=n_topics,
                    passes=passes,
                    id2word={i: s for i, s in enumerate(vocabulary)})


if __name__ == '__main__':
    content, input_type, tokenizer = dataset_mails(
        '/Users/yanchith/workspace/won-corpora/processed')

    # content, input_type, tokenizer = dataset_newsgroups()

    # tokenizer=tokenizer
    vectorizer = TfidfVectorizer(min_df=5, input=input_type, ngram_range=(1, 1),
                                 stop_words='english', tokenizer=tokenizer)
    X = vectorizer.fit_transform(content)
    features = vectorizer.get_feature_names()

    print(len(features))
    print(X.shape)

    print(features)

    topics = 200

    lda_model = fit_lda(X.T, features, n_topics=topics)
    for topic in lda_model.show_topics(num_topics=topics, formatted=True):
        print(topic)

