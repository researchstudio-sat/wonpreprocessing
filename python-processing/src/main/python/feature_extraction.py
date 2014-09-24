import numpy as np
from scipy.sparse.coo import coo_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import WordNetLemmatizer

import six

_wnl = WordNetLemmatizer()

_filter_threshold = np.vectorize(lambda x, threshold: x if x > threshold else 0,
                                 excluded={'threshold'})


def _clean_mail(doc):
    """Return only the Subject and Content parts from the mail."""
    subject, content = False, False
    accepted_lines = []
    for line in doc.splitlines(keepends=True):
        l = line.lower()
        if l.startswith('subject'):
            subject, content = True, False
            line = line.lstrip('subject')
        elif l.startswith('content'):
            subject, content = False, True
            line = line.lstrip('content')
        if subject or content:
            accepted_lines.append(line.strip())
    return ''.join(accepted_lines)


def lemma_tokenizer(doc):
    return [_wnl.lemmatize(t) for t in nltk.word_tokenize(_clean_mail(doc))]


class PosTagLemmaTokenizer:
    def __init__(self):
        self.tags = {'NN', 'NNS', 'VBZ', 'JJ', 'RB', 'VBG'}
        self.tagger = nltk.data.load(nltk.tag._POS_TAGGER)

    def __call__(self, doc):
        tokens = nltk.word_tokenize(_clean_mail(doc))
        tagged = self.tagger.tag(tokens)
        return [_wnl.lemmatize(t[0]) for t in tagged if t[1] in self.tags]


def vectorize_and_transform(filenames, tokenizer=None):
    if tokenizer is None:
        tokenizer = lemma_tokenizer

    vectorizer = TfidfVectorizer(min_df=2, stop_words='english',
                                 input='filename', ngram_range=(1, 2),
                                 tokenizer=tokenizer)
    fit = vectorizer.fit_transform(filenames)
    return fit, vectorizer.get_feature_names()


def apply_threshold(original, threshold):
    original = coo_matrix(original)  # Ensure COO format
    indices = original.data > threshold
    new_data = original.data[indices]
    new_matrix = new_data, (original.row[indices], original.col[indices])
    return coo_matrix(new_matrix)

# Run an example
if __name__ == '__main__':
    import sys
    from os import listdir
    from os.path import isfile, join

    if len(sys.argv) != 2:
        raise Exception("USAGE: python feature_extraction.py <input dir>")

    path = sys.argv[1]  # 0-th argument is the script name.
    files = [join(path, f) for f in listdir(path) if
             isfile(join(path, f)) and f.endswith('.eml')]

    print('Loaded ', len(files), ' files from path: ', path)

    data, features = vectorize_and_transform(files)

    above_threshold = apply_threshold(data, 0.3)

    document_indices = above_threshold.row
    keyword_indices = above_threshold.col

    print('\n\nAccepted documents: ', len(np.unique(document_indices)))

    last_di = -1
    for di, ki in zip(document_indices, keyword_indices):
        if di != last_di:
            print('\n\n', files[di])
            last_di = di
        six.print_(features[ki], end=' | ')