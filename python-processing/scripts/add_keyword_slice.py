import codecs
import sys

import numpy as np
from scipy.sparse.coo import coo_matrix
from scipy.io import mmwrite
import six

from tools.feature_extraction import apply_threshold, new_tensor_slice, \
    create_vectorizer
from tools.datasets import dataset_mails, dataset_small


if len(sys.argv) < 2:
    raise Exception('ARGS: <documents dir> <rescal dir>')

doc_path = sys.argv[1]
rescal_path = sys.argv[2]


def get_document_names(path_prefix, file_paths):
    to_ignore = set()
    documents = []
    for i, file_path in enumerate(file_paths):
        try:  # Upon manipulation we might find something strange.
            document_name = six.text_type(
                file_path.rstrip('.eml').lstrip(path_prefix))
            documents.append(document_name)
        except UnicodeDecodeError:  # So we will not use that file.
            to_ignore.add(i)
            print('Skipping file: ', file_path, '.')
    for i in to_ignore:  # Remove incorrect documents also from file paths.
        del file_paths[i]
    return documents, file_paths


print('Loading documents.')

# paths, input_type, tokenizer = dataset_small()
# docs = list(paths)

paths, input_type, tokenizer = dataset_mails(doc_path)
docs, paths = get_document_names(doc_path, paths)

print('Loaded ', len(docs), ' files from path: ', doc_path, '.')

print('Extracting features.')
vectorizer = create_vectorizer(input_type, tokenizer=tokenizer, min_df=2)

data = vectorizer.fit_transform(paths)
features = vectorizer.get_feature_names()

data = apply_threshold(data, 0.2)

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

print('Writing headers.')

with codecs.open(rescal_path + '/headers.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(new_headers))

print('Writing keywords.')
mmwrite(rescal_path + '/keywords_slice.mtx', offset_matrix)


