import sys
from os import listdir
from os.path import isfile, join

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


if len(sys.argv) != 2:
    raise Exception("USAGE: python feature_extraction.py <input dir>")

path = sys.argv[1]  # 0-th argument is the script name.

print('Loading corpus documents from ', path)

filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
print('Loaded ', len(filenames), ' files')

print('Vectorizing...', end=' ')

vectorizer = TfidfVectorizer(min_df=2, stop_words='english', input='filename',
                             ngram_range=(1, 2))
fit = vectorizer.fit_transform(filenames)

print('DONE')

feature_names = vectorizer.get_feature_names()
print('Features: ', feature_names)

threshold = 0.25
print('Relevancy threshold: ', threshold)

above_threshold = np.nonzero(fit > threshold)

document_indices = above_threshold[0]
keyword_indices = above_threshold[1]

print('Keywords:')
last_di = -1
for di, ki in zip(document_indices, keyword_indices):
    if di != last_di:
        print('\n\n', filenames[di])
        last_di = di
    print(feature_names[ki], end=' ')

print('\n\nAccepted documents: ', len(np.unique(document_indices)))
