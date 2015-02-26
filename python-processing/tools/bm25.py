__author__ = 'hfriedrich'

import numpy as np
from tools.tensor_utils import SparseTensor
from math import log10
from scipy.sparse import csr_matrix

# see http://en.wikipedia.org/wiki/Okapi_BM25
# parameters:
# tensor: SparseTensor object
# indices: indices pointing to (need, need) combinations to compute the connection bm25 score for
# threshold: if threshold is given result array is binary
# var_k: see http://en.wikipedia.org/wiki/Okapi_BM25
# var_b: http://en.wikipedia.org/wiki/Okapi_BM25
# return: array with result bm25 scores (float or binary) for each index pair
def bm25_link_prediciton(tensor, indices, threshold=None, var_k=1.5, var_b=0.75):

    # combine the attributes in one matrix
    m_csr = m = tensor.getSliceMatrix(SparseTensor.ATTR_SUBJECT_SLICE) + \
                tensor.getSliceMatrix(SparseTensor.ATTR_CONTENT_SLICE) + \
                tensor.getSliceMatrix(SparseTensor.CATEGORY_SLICE)

    # compute the average document length
    numNeeds = len(tensor.getNeedIndices())
    avgDocLength = len(m_csr[tensor.getNeedIndices(),:].nonzero()[1]) / float(numNeeds)

    # create a dictionary with computed idf values of all attributes
    idf = dict()
    m_csc = m_csr.tocsc()
    for attr in tensor.getAttributeIndices():
        n = len(m_csc[:,attr].nonzero()[1])
        idf[attr] = log10((numNeeds - n + 0.5) / (n + 0.5))

    # compute the BM25 score for every index
    # if a threshold is specified use it to set the result prediction value to 0 or 1
    prediction = []
    for i in range(len(indices[0])):
        docNeed = indices[0][i]
        docLength = len(m[docNeed,:].nonzero()[1])
        queryNeed = indices[1][i]
        queryAttrs = m_csr[queryNeed,:].nonzero()[1]
        s = 0
        if len(m[docNeed,queryAttrs].nonzero()[0]) > 0:
            queryAttrs = [attr for attr in queryAttrs if m[docNeed,attr] != 0.0]
            if len(queryAttrs) > 0:
                s = score(docLength, avgDocLength, queryAttrs, idf, var_k, var_b)
            if threshold != None:
                s = 1 if s > threshold else 0
        prediction.append(s)
    return prediction

# for computation of the score use term frequency either 0 or 1 since we don't have the number of occurrences of an
# attribute in a need. This is why in queryAttrs only are attributes that actually are part of the document.
def score(docLength, avgDocLength, queryAttrs, idf, k, b):
    sum = 0
    d = (k + 1) / (1 + k * (1 - b + b * docLength / avgDocLength))
    for attr in queryAttrs:
        sum += idf[attr] * d
    return sum



