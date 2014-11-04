from scipy.sparse.coo import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize


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


def create_vectorizer(input_type, tokenizer=None, min_df=1,
                      stop_words='english', ngram_range=(1, 1)):
    return TfidfVectorizer(input=input_type, tokenizer=tokenizer, min_df=min_df,
                           stop_words=stop_words, ngram_range=ngram_range)


def vectorize_and_transform(filenames, tokenizer=None, vectorizer=None,
                            input_type='filename'):
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
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english',
                                     input=input_type, ngram_range=(1, 2),
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


def new_tensor_slice(headers, feature_definitions):
    """
    Returns updated headers definition with the indices for creating a new
    coo matrix. Features not used are discarded together with their indices.

    :param headers: The original headers. Will not be modified.
    :param feature_definitions: The 'dimension' definitions. Iterable of
    ("dimension name", [feature values])
    :return: The updated headers, new indices for current features
    """
    headers = list(headers)

    assert len(feature_definitions) == 2

    used_sets = {}
    data_offset_indices = {}

    column_names = map(lambda x: x[0], feature_definitions)
    for name, _, row_or_col in feature_definitions:
        data_offset_indices[name] = {}
        used_sets[name] = set(row_or_col)

    # Read headers for existing docs, features.
    headers_cursor = 1

    for header in headers:
        for name in column_names:
            column_data_index = data_offset_indices[name]
            if header.startswith(name):
                suffix = header[len(name):]
                column_data_index[suffix] = headers_cursor
                break
        headers_cursor += 1

    for name, values, _ in feature_definitions:
        column_data_index = data_offset_indices[name]
        column_used_set = used_sets[name]
        for i, data_point in enumerate(values):
            if i not in column_used_set:
                continue
            if data_point not in column_data_index:
                headers.append(name + data_point)
                column_data_index[data_point] = headers_cursor
                headers_cursor += 1

    return headers, data_offset_indices