from scipy.io.mmio import mmwrite

__author__ = 'hfriedrich'

import logging
import codecs
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from rescal import rescal_als

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
_log = logging.getLogger('tensor utils')

# This file contains util functions for the processing of the tensor (including handling
# of needs, attributes, etc.)

CONNECTION_SLICE = 0
NEED_TYPE_SLICE = 1
ATTR_SUBJECT_SLICE = 2



# read the input tensor data (e.g. data-0.mtx ... data-3.mtx) and
# the headers file (e.g. headers.txt)
# if adjustDim is True then the dimensions of the slice matrix
# files are automatically adjusted to fit to biggest dimensions of all slices
def read_input_tensor(headers_filename, data_file_names, adjustDim=False):

    #load the header file
    _log.info("Read header input file: " + headers_filename)
    input = codecs.open(headers_filename,'r',encoding='utf8')
    headers = input.read().splitlines()
    input.close()

    # get the largest dimension of all slices
    if adjustDim:
        maxDim = 0
        for data_file in data_file_names:
            matrix = mmread(data_file)
            if maxDim < matrix.shape[0]:
                maxDim = matrix.shape[0]
            if maxDim < matrix.shape[1]:
                maxDim = matrix.shape[1]

    # load the data files
    K = []
    slice = 0
    for data_file in data_file_names:
        if adjustDim:
            _log.warn("Adujst dimension to (%d,%d) of matrix file: %s" % (maxDim, maxDim, data_file))
            adjust_mm_dimension(data_file, maxDim)
        _log.info("Read as slice %d the data input file: %s" % (slice, data_file))
        matrix = mmread(data_file)
        if slice == 0:
            dims = matrix.shape
        if dims != matrix.shape or matrix.shape[0] != len(headers) or matrix.shape[0] != matrix.shape[1]:
            raise Exception("Bad shape of input slices of tensor!")
        dims = matrix.shape
        K.append(csr_matrix(matrix))
        slice = slice + 1
    return K, headers

# adjust (increase) the dimension of an mm matrix file
def adjust_mm_dimension(data_file, dim):
    file = codecs.open(data_file,'r',encoding='utf8')
    lines = file.read().splitlines()
    file.close()
    file = codecs.open(data_file,'w+',encoding='utf8')
    found = False
    for line in lines:
        if not line.startswith('%') and not found:
            vals = line.split(' ')
            newLine = str(dim) + " " + str(dim) + " " + vals[2]
            file.write(newLine + "\n")
            found = True
        else:
            file.write(line + "\n")
    file.close()

# return a tuple with two lists holding need indices that represent connections
# between these needs, symmetric connection are only represented once
def connection_indices(tensor):
    nz = tensor[CONNECTION_SLICE].nonzero()
    nz0 = [nz[0][i] for i in range(len(nz[0])) if nz[0][i] <= nz[1][i]]
    nz1 = [nz[1][i] for i in range(len(nz[0])) if nz[0][i] <= nz[1][i]]
    indices = [i for i in range(len(nz0))]
    np.random.shuffle(indices)
    ret0 = [nz0[i] for i in indices]
    ret1 = [nz1[i] for i in indices]
    nzsym = (ret0, ret1)
    return nzsym

# return a list of indices which refer to rows/columns of needs in the tensor
def need_indices(headers):
    needs = [i for i in range(0, len(headers)) if (headers[i].startswith('Need:'))]
    return needs

# return a list of indices which refer to rows/columns of needs in the tensor that were checked manually for
# connections to other needs.
def manually_checked_needs(headers, connection_file):
    _log.info("Read connections input file: " + connection_file)
    file = codecs.open(connection_file,'r',encoding='utf8')
    lines = file.read().splitlines()
    checked_need_names = ["Need: " + lines[0]] + ["Need: " + lines[i] for i in range(1,len(lines))
                                                  if lines[i-1] == "" and lines[i] != ""]
    checked_needs = [i for i in need_indices(headers) if headers[i] in checked_need_names]
    file.close()
    return checked_needs

# return a list of indices which refer to rows/columns of needs of type OFFER in the tensor
def offer_indices(tensor, headers):
    needs = need_indices(headers)
    offer_attr_idx = headers.index("Attr: OFFER")
    offers = [need for need in needs if (tensor[NEED_TYPE_SLICE][need, offer_attr_idx] == 1)]
    return offers

# return a list of indices which refer to rows/columns of needs of type WANT in the tensor
def want_indices(tensor, headers):
    needs = need_indices(headers)
    want_attr_idx = headers.index("Attr: WANT")
    wants = [need for need in needs if (tensor[NEED_TYPE_SLICE][need, want_attr_idx] == 1)]
    return wants

# execute the rescal algorithm and return a prediction tensor
def predict_rescal_als(input_tensor, rank):
    _log.info('start rescal processing ...')
    _log.info('Datasize: %d x %d x %d | Rank: %d' % (
        input_tensor[0].shape + (len(input_tensor),) + (rank,))
    )
    A, R, _, _, _ = rescal_als(
        input_tensor, rank, init='nvecs', conv=1e-3,
        lambda_A=0, lambda_R=0, compute_fit='true'
    )
    n = A.shape[0]
    P = np.zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = np.dot(A, np.dot(R[k], A.T))
    return P, A, R

# create a similarity matrix of needs (and attributes)
def similarity_ranking(A):
    dist = squareform(pdist(A, metric='cosine'))
    return dist

# for rescal algorithm output predict connections by fixed threshold for each of the test_needs based on the
# similarity of latent need clusters (higher threshold means higher recall)
def predict_rescal_connections_by_need_similarity(P, A, threshold, all_offers, all_wants, test_needs):
    S = similarity_ranking(A)
    binary_prediction = np.zeros(P.shape)
    for need in test_needs:
        if need in all_offers:
            all_needs = all_wants
        elif need in all_wants:
            all_needs = all_offers
        else:
            continue

        for x in all_needs:
            if S[need,x] < threshold:
                binary_prediction[need, x, CONNECTION_SLICE] = 1
    return binary_prediction

# for rescal algorithm output predict connections by fixed threshold (higher threshold means higher precision)
def predict_rescal_connections_by_threshold(P, threshold, all_offers, all_wants, test_needs):
    binary_prediction = np.zeros(P.shape)
    for need in test_needs:
        if need in all_offers:
            all_needs = all_wants
        elif need in all_wants:
            all_needs = all_offers
        else:
            continue
        for x in all_needs:
            if P[need,x,CONNECTION_SLICE] >= threshold:
                binary_prediction[need,x,CONNECTION_SLICE] = 1
    return binary_prediction

