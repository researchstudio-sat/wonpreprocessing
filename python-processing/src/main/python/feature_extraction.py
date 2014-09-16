import sys
from os import listdir
from os.path import isfile, join

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer


class PosTagLemmaTokenizer(object):
    def __init__(self):
        self.tags = {'NN', 'NNS', 'VBZ', 'JJ', 'RB', 'VBG'}
        self.tagger = nltk.data.load(nltk.tag._POS_TAGGER)
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        subject, content = False, False
        accepted_lines = []
        for line in doc.splitlines(keepends=True):
            l = line.lower()
            if l.startswith('subject'):
                subject = True
                line = line.lstrip('subject')
            elif l.startswith('content'):
                subject, content = False, True
                line = line.lstrip('content')
            if subject or content:
                accepted_lines.append(line.strip())
        doc = ''.join(accepted_lines)
        tokens = nltk.word_tokenize(doc)
        # return [self.wnl.lemmatize(t) for t in tokens]
        tagged = self.tagger.tag(tokens)
        return [self.wnl.lemmatize(t[0]) for t in tagged if t[1] in self.tags]

vectorizer = TfidfVectorizer(min_df=2, stop_words='english', input='filename',
                             ngram_range=(1, 2), tokenizer=PosTagLemmaTokenizer())


def vectorize_and_filter(filenames, threshold):
    fit = vectorizer.fit_transform(filenames)
    return fit, np.nonzero(fit > threshold)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception("USAGE: python feature_extraction.py <input dir>")

    path = sys.argv[1]  # 0-th argument is the script name.
    print('Loading corpus documents from ', path)

    files = [join(path, f) for f in listdir(path) if
             isfile(join(path, f)) and f.endswith('.eml')]

    print('Loaded ', len(files), ' files')

    data, above_threshold = vectorize_and_filter(files, 0.3)
    print(data)
    print()
    print(data.shape)
    print(data[0, 0])

    document_indices = above_threshold[0]
    keyword_indices = above_threshold[1]

    feature_names = vectorizer.get_feature_names()
    print('Features: ', feature_names)

    print('Keywords:')
    last_di = -1
    for di, ki in zip(document_indices, keyword_indices):
        if di != last_di:
            print()
            print()
            print(files[di])
            last_di = di
        print(feature_names[ki], end=' | ')
    print('\n\nAccepted documents: ', len(np.unique(document_indices)))
    # tensor, headers = evaluate_link_prediction.read_input_tensor(
    # path + '/rescal')
    # filemap = {}
    # index = 1
    # for line in headers:
    # if line.startswith('Need: '):
    #         filemap[line[6:]] = index
    #     index += 1
    #
    #
    #
    #     # TODO: map current document indices to new -> filemap[files[di]]
    #     # TODO: map current feature indices to new -> fi + n_headers
    #
    #     # for di, ki in zip(document_indices, keyword_indices):
    #     # filemap[files[di]]





