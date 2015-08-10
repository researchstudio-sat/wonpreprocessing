__author__ = 'Federico'
# Multilabel (i.e. a sample is assigned to more than one category) Naive Bayes classifier for WoN dataset
#It uses OneVsRest, MultinomialNB classification strategies

from numpy import *
from tools.tensor_utils import read_input_tensor, SparseTensor
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from sklearn.multiclass import OneVsRestClassifier

def get_example_data():

    # read the tensor from the folder passed by args
    data_file_prefix = sys.argv[1]
    header_file = data_file_prefix + '/headers.txt'
    data_files = [data_file_prefix + "/connection.mtx",
                  data_file_prefix + "/needtype.mtx",
                  data_file_prefix + "/subject.mtx",
                  data_file_prefix + "/content.mtx",
                  data_file_prefix + "/category.mtx"]
    slices = [SparseTensor.CONNECTION_SLICE, SparseTensor.NEED_TYPE_SLICE, SparseTensor.ATTR_SUBJECT_SLICE,
              SparseTensor.ATTR_CONTENT_SLICE, SparseTensor.CATEGORY_SLICE]

    tensor = read_input_tensor(header_file, data_files, slices, False)

    data = []
    target = []

    # Store the chosen input into lists.
    for need_index in tensor.getNeedIndices():
        content = ""
        category_index = tensor.getSliceMatrix(SparseTensor.CATEGORY_SLICE)[need_index,].nonzero()[1].tolist()
        target.append(category_index)
        for word in tensor.getAttributesForNeed(need_index, SparseTensor.ATTR_SUBJECT_SLICE):
            content += word + " "
        data.append(content)

    # Print out the input, just a check:
    target_names = tensor.getHeaders()
    print("test")
    print data
    print target_names
    print target

    return data, target, target_names

# Call for the input
my_data, my_target, my_targetname = get_example_data()

# A little information about dimensions and format of the input:
print type(my_data), type(my_target),   # format of data and targets
print len(my_data)  # number of samples
print len(my_target)


# Let's build the training and testing datasets:
SPLIT_PERC = 0.80  # 80% goes into training, 20% into test
split_size = int(len(my_data)*SPLIT_PERC)
X_train = my_data[:split_size]
X_test = my_data[split_size:]
y_train = my_target[:split_size]
y_test = my_target[split_size:]


# Training, prediction and evaluation of the classifier(s):
def train_and_evaluate(clf, X_train, X_test, y_train, y_test, y_name):
    #Training and prediction
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Precision, recall and support (i.e. nr. of samples used for the testing)
    print "\n Classification Report: \n"
    print metrics.classification_report(y_test, y_pred)

# Introducing stop words
stopset = set(stopwords.words('english'))

# Two different classifiers: Count, Tfidf vectorization
clf_count = Pipeline([
    ('vect', CountVectorizer(
        stop_words=stopset,
        token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.01))),
    ])

clf_tfidf = Pipeline([
    ('vect', TfidfVectorizer(
        stop_words=stopset,
        token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.01))),
    ])

# List of classifiers
clfs = [clf_count, clf_tfidf]

# Run the evaluation/classification
for clf in clfs:
    train_and_evaluate(clf, X_train, X_test, y_train, y_test, my_targetname)
