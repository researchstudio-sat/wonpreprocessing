from gensim import matutils
from gensim.models import LdaModel
from gensim.models.hdpmodel import HdpModel
from sklearn.feature_extraction.text import TfidfVectorizer

from tools.datasets import dataset_mails

# TODO: move to scripts


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

