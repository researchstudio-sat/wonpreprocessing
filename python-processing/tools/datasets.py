from os import listdir
from os.path import join, isfile

from nltk import WordNetLemmatizer
import numpy as np
from sklearn.datasets import fetch_20newsgroups

from tools.feature_extraction import ScikitNltkTokenizerAdapter, \
    default_pos_tagger
from tools.mails import mail_preprocessor


def dataset_newsgroups(categories=None):
    random = np.random.RandomState()
    data = fetch_20newsgroups(
        categories=categories, subset='train', shuffle=True,
        random_state=random, remove=('headers', 'footers', 'quotes'))
    # pos_tagger=default_pos_tagger
    tokenize = ScikitNltkTokenizerAdapter(
        lemmatizer=WordNetLemmatizer(), pos_tagger=default_pos_tagger)
    return data.data, 'content', tokenize


def dataset_mails(path):
    filenames = [join(path, f) for f in listdir(path) if
                 isfile(join(path, f)) and f.endswith('.eml')]
    filenames = np.array(filenames)
    np.random.shuffle(filenames)
    # pos_tagger=default_pos_tagger
    tokenize = ScikitNltkTokenizerAdapter(preprocessor=mail_preprocessor,
                                          lemmatizer=WordNetLemmatizer())
    return filenames, 'filename', tokenize


def dataset_small():
    content = ["This is the first sentence. Here is another sentence! And here's a third sentence.",
               "This is the second paragraph.",
               "Tokenization is currently fairly simple, so the period in Mr. gets tokenized."]
    np.random.shuffle(content)
    return content, 'content', ScikitNltkTokenizerAdapter(
        lemmatizer=WordNetLemmatizer())