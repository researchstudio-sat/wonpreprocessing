import codecs
import sys
from os import listdir
from os.path import isfile, join

from nltk.stem.wordnet import WordNetLemmatizer

import numpy as np
from scipy.sparse.coo import coo_matrix
from scipy.io import mmwrite
import six

from feature_extraction import vectorize_and_transform, apply_threshold, \
    ScikitNltkTokenizerAdapter, default_tagger_and_tags
from mail_utils import mail_preprocessor

if len(sys.argv) < 3:
    raise Exception('ARGS: <documents dir> <rescal dir> [<skip pos tagging>]')

doc_path = sys.argv[1]
rescal_path = sys.argv[2]
if len(sys.argv) == 4 and sys.argv[3].lower() in {'y', 'yes', 't', 'true'}:
    print('Will not use POS tagging')
    pos_tagger = None
else:
    print('Will use POS tagging.')
    pos_tagger = default_tagger_and_tags

tokenizer = ScikitNltkTokenizerAdapter(preprocessor=mail_preprocessor,
                                       tagger_and_tags=pos_tagger,
                                       lemmatizer=WordNetLemmatizer())

print('Loading documents.')

documents = []
for f in listdir(doc_path):
    if isfile(join(doc_path, f)) and f.endswith('.eml'):
        try:
            documents.append(six.text_type(f.rstrip('.eml')))
        except UnicodeDecodeError:
            print('Skipping file: ', f, '.')

print('Loaded ', len(documents), ' files from path: ', doc_path, '.')

print('Extracting features.')

file_paths = [join(doc_path, f + '.eml') for f in documents]
data, features = vectorize_and_transform(file_paths, tokenizer=tokenizer)
data = apply_threshold(data, 0.3)

# TODO: filter features and filenames based on whether their entries survived

print('Reading headers.')
with codecs.open(rescal_path + '/headers.txt', 'r', encoding='utf8') as f:
    headers = f.read().splitlines()


NEED_STR = 'Need: '
ATTR_STR = 'Attr: '
LEN = len(NEED_STR)

document_index, feature_index = {}, {}
headers_cursor = 1
for header in headers:
    if header.startswith(NEED_STR):
        document_index[header[LEN:]] = headers_cursor
    elif header.startswith(ATTR_STR):
        feature_index[header[LEN:]] = headers_cursor
    headers_cursor += 1

for filename in documents:
    if filename not in document_index:
        # print('Adding NEED at index (', headers_cursor, '): ', filename)
        headers.append(NEED_STR + filename)
        document_index[filename] = headers_cursor
        headers_cursor += 1

for feature in features:
    if feature not in document_index:
        # print('Adding ATTR at index (', headers_cursor, '): ', feature)
        headers.append(ATTR_STR + feature)
        feature_index[feature] = headers_cursor
        headers_cursor += 1
    else:
        print('Feature reused: ', feature)

print('Offsetting slice.')

offset_row = np.array([document_index[documents[i]] for i in data.row])
offset_col = np.array([feature_index[features[i]] for i in data.col])

offset_matrix = coo_matrix((data.data, (offset_row, offset_col)))

print('Writing headers.')

with codecs.open(rescal_path + '/headers.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(headers))

mmwrite(rescal_path + '/keywords_slice.mtx', offset_matrix)


