from os import listdir
from os.path import isfile, join

from gensim import matutils
from gensim.models import LdaModel
from gensim.models.hdpmodel import HdpModel
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
    # pos_tagger=default_pos_tagger
    tokenize = ScikitNltkTokenizerAdapter(lemmatizer=WordNetLemmatizer(),
                                          pos_tagger=default_pos_tagger)
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


def fit_lda(corpus, vocabulary, n_topics=10, passes=1):
    return LdaModel(corpus, num_topics=n_topics, passes=passes,
                    id2word={i: s for i, s in enumerate(vocabulary)})


def fit_hdp_lda(corpus, vocabulary):
    return HdpModel(corpus, {i: s for i, s in enumerate(vocabulary)})


if __name__ == '__main__':
    content, input_type, tokenizer = dataset_mails(
        '/Users/yanchith/workspace/won-corpora/processed')

    # content, input_type, tokenizer = dataset_newsgroups()

    vectorizer = TfidfVectorizer(min_df=3, input=input_type, ngram_range=(1, 1),
                                 stop_words='english', tokenizer=tokenizer)
    X = vectorizer.fit_transform(content)
    features = vectorizer.get_feature_names()

    print('Number of features:', len(features))
    print('Bag of words shape:', X.shape)
    print(features)

    # Beware, gensim requires the matrix transposed
    model = fit_hdp_lda(matutils.Sparse2Corpus(X, documents_columns=False),
                        features)

    n_topics_to_show = 200
    for topic in model.show_topics(topics=n_topics_to_show, topn=10,
                                   formatted=True):
        print(topic)

