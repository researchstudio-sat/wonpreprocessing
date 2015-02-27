import codecs
import sys

import numpy as np
from scipy.sparse.coo import coo_matrix
from scipy.io import mmwrite
import six

from tools.feature_extraction import apply_threshold, new_tensor_slice, \
    create_vectorizer
from tools.datasets import dataset_mails, dataset_small

YES = {'y', 'yes', 't', 'true'}

if len(sys.argv) < 2:
    raise Exception('ARGS: <documents dir> <rescal dir>')

use_small_dataset = len(sys.argv) > 3 and sys.argv[3].lower() in YES
doc_path = sys.argv[1]
rescal_path = sys.argv[2]


def get_document_names(path_prefix, file_paths):
    documents = []
    new_file_paths = file_paths[:]
    for i, file_path in enumerate(file_paths):
        try:  # Upon manipulation we might find something strange.
            document_name = six.text_type(
                file_path.rstrip('.eml').lstrip(path_prefix))
            documents.append(document_name)
        except UnicodeDecodeError:  # So we will not use that file.
            del new_file_paths[i]
            print('Skipping file: ', file_path, '.')
    return documents, new_file_paths


print('Loading documents.')

if use_small_dataset:
    docs, input_type, tokenizer = dataset_small()
    paths = docs
else:
    unchecked_paths, input_type, tokenizer = dataset_mails(doc_path)
    docs, paths = get_document_names(doc_path, unchecked_paths)

print('Loaded ', len(docs), ' files from path: ', doc_path, '.')

print('Extracting features.')
vectorizer = create_vectorizer(input_type, tokenizer=tokenizer,
                               ngram_range=(1, 1))

data = vectorizer.fit_transform(paths)
features = vectorizer.get_feature_names()

data = coo_matrix(data)

data = apply_threshold(data, 0.1)  # Filter out everything, that is too weak.

print('Reading headers.')
with codecs.open(rescal_path + '/headers.txt', 'r', encoding='utf8') as f:
    original_headers = f.read().splitlines()

DOCUMENT = 'Need: '
FEATURE = 'Attr: '

new_headers, offsets = new_tensor_slice(original_headers, [
    (DOCUMENT, docs, data.row), (FEATURE, features, data.col)]
)

document_offsets = offsets[DOCUMENT]
feature_offsets = offsets[FEATURE]

print('Offsetting slice.')

offset_row = np.array([document_offsets[docs[d]] - 1 for d in data.row])
offset_col = np.array([feature_offsets[features[d]] - 1 for d in data.col])

print('Contrasting slice.')
contrasted_data = (data.data > 0).astype(float)

offset_matrix = coo_matrix((contrasted_data, (offset_row, offset_col)))
float_matrix = coo_matrix((data.data, (offset_row, offset_col)))

print('Writing headers.')

with codecs.open(rescal_path + '/headers.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(new_headers))

print('Writing keywords.')
mmwrite(rescal_path + '/keyword.mtx', offset_matrix)
mmwrite(rescal_path + '/keyword_float.mtx', float_matrix)


