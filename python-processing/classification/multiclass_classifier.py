__author__ = 'Federico'
# Multiclass Naive-Bayes classifier for categorization of WoN e-mail dataset
# It uses MultinomialNB classifier

from numpy import *
from tools.tensor_utils import read_input_tensor, SparseTensor
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords

# Get the input from a folder in C:
def get_example_data():

    header_file = 'C:/Users/Federico/Desktop/test/evaluation/tensor_content_NEW/headers.txt'
    data_file_prefix = 'C:/Users/Federico/Desktop/test/evaluation/tensor_content_NEW'
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
    # The "if" statement is meant to include only samples with a single category (No multilabel)
    for need_index in tensor.getNeedIndices():
        content = ""
        categories = tensor.getAttributesForNeed(need_index, SparseTensor.CATEGORY_SLICE)
        numCategories = len(categories)
        if numCategories >= 1:
            category_index = tensor.getSliceMatrix(SparseTensor.CATEGORY_SLICE)[need_index,].nonzero()[1][0]
            target.append(category_index)
            for word in tensor.getAttributesForNeed(need_index, SparseTensor.ATTR_SUBJECT_SLICE):
                content += word + " "
            data.append(content)

    # Include only few of all the categories (e.g. with samples > n)
    newdata = []
    newtarget = []
    for i in range(len(target)):

        if target.count(target[i]) > 50:
            newtarget.append(target[i])
            newdata.append(data[i])

    data = newdata
    target = newtarget

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

    # Training
    clf.fit(X_train, y_train)
    # Prediction of testing sets
    y_pred = clf.predict(X_test)

    # Precision, recall and support (i.e. nr. of samples used for the testing)
    print "Classification Report:"
    print metrics.classification_report(y_test, y_pred)
    # Confusion Matrix
    print "Confusion Matrix:"
    print metrics.confusion_matrix(y_test, y_pred)

    # Visualization of Categories / Assigned / Data
    print "Tested data => assigned category,    data:"
    for i in range(len(X_test)):
        print str(i) + ")   Real category: " + str(y_name[y_test[i]]) + ",    Assigned category: " + \
            str(y_name[y_pred[i]]) + ",     Data: " + str(X_test[i])

    # Assign names to the categories (defined by numbers)
    print "\n Categories: \n"
    categories = set()
    for cat in y_pred:
        categories.add(cat)
    categories = sorted(categories)
    for cat in categories:
        print str(cat) + "    " + y_name[cat]

# Introducing stop words
stopset = set(stopwords.words('english'))

# Two different classifiers: Count and Tfidf vectors
clf_count = Pipeline([
    ('vect', CountVectorizer(
        stop_words=stopset,
        token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB(alpha=1)),
    ])

clf_tfidf = Pipeline([
     ('vect', TfidfVectorizer(
         stop_words=stopset,
         token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
     )),
     ('clf', MultinomialNB(alpha=1)),
     ])

# List of classifiers
clfs = [clf_count, clf_tfidf]

# Run the evaluation/classification
for clf in clfs:
    train_and_evaluate(clf, X_train, X_test, y_train, y_test, my_targetname)
