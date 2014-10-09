import numpy as np
from scipy.sparse.coo import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import six

from mail_utils import mail_preprocessor


_stop_symbols = {'(', ')', '<', '>', '[', ']'}

_default_tagger = nltk.data.load(nltk.tag._POS_TAGGER)
_default_tags = {'NN', 'NNS', 'VBZ', 'JJ', 'RB', 'VBG'}

default_pos_tagger = (_default_tagger, _default_tags)


class ScikitNltkTokenizerAdapter:
    def __init__(self, preprocessor=None, tokenizer=None, pos_tagger=None,
                 lemmatizer=None):
        self.preprocessor = preprocessor
        if tokenizer is None:
            self.tokenizer = word_tokenize
        else:
            self.tokenizer = tokenizer
        if pos_tagger is not None:
            self.tagger, self.tags = pos_tagger
        else:
            self.tagger, self.tags = None, None
        self.lemmatizer = lemmatizer

    def __call__(self, doc):
        if self.preprocessor is not None:
            doc = self.preprocessor(doc)
        tokens = self.tokenizer(doc)
        if self.tagger is not None:
            tagged = self.tagger.tag(tokens)
            tokens = [token for token, tag in tagged if tag in self.tags]
        if self.lemmatizer is None:
            return tokens
        return [self.lemmatizer.lemmatize(token) for token in tokens]


def vectorize_and_transform(filenames, tokenizer=None, vectorizer=None):
    """
    Extract the bag-of-words model from the provided files.

    :param filenames: Iterable of absolute paths to documents
    :param tokenizer: The tokenizer to plug into the vectorizer
    :param vectorizer: The vectorizer to use (tokenizer will be ignored).
    :return: tf-idf matrix, vocabulary
    """
    if vectorizer is None:
        if tokenizer is None:
            tokenizer = ScikitNltkTokenizerAdapter()
        vectorizer = TfidfVectorizer(min_df=2, stop_words='english',
                                     input='filename', ngram_range=(1, 2),
                                     tokenizer=tokenizer)

    fit = vectorizer.fit_transform(filenames)
    return fit, vectorizer.get_feature_names()


def apply_threshold(original, threshold):
    """
    Filter the matrix such that each field has is greater than the threshold.

    :param original: The original matrix.
    :param threshold: Numeric threshold applied to each cell.
    :return: COO matrix with cells filtered according to the threshold.
    """
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

    adapter = ScikitNltkTokenizerAdapter(preprocessor=mail_preprocessor,
                                         pos_tagger=default_pos_tagger,
                                         lemmatizer=WordNetLemmatizer())
    data, features = vectorize_and_transform(files, adapter)

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