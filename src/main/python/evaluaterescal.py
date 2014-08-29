#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
_log = logging.getLogger('Mail Example')

import sys
import numpy as np
from numpy import dot, array, zeros, setdiff1d
from numpy.linalg import norm
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix, csr_matrix
from scipy import sparse
from rescal import rescal_als
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.io import mmwrite
import codecs
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, confusion_matrix
import random
from numpy.random import shuffle

def predict_rescal_als(A, R):
    n = A.shape[0]
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return P

def similarity_ranking(A):
    dist = squareform(pdist(A, metric='cosine'))
    return dist

def read_input_tensor(data_file, header_file):
    _log.info("Read mat input file: " + data_file)
    mat = loadmat(data_file)
    _log.info("Read header input file: " + header_file)
    input = codecs.open(header_file,'r',encoding='utf8')
    headers = input.read().splitlines()
    K = []
    for i in range(0,3):
        G = mat['Rs' + str(i)]
        I,J = G.nonzero()
        V = G.data
        K.append(sparse.csr_matrix((V,(I,J)),G.shape))
    return K, headers

def connection_indices(tensor):
    nz = tensor[0].nonzero()
    nz0 = [nz[0][i] for i in range(len(nz[0])) if nz[0][i] <= nz[1][i]]
    nz1 = [nz[1][i] for i in range(len(nz[0])) if nz[0][i] <= nz[1][i]]
    nzsym = (nz0, nz1)
    return nzsym

def need_indices(headers):
    needs = [i for i in range(0, len(headers)) if (headers[i].startswith('Need:'))]
    return needs

def fullrow_need_connection_indices(tensor, need_indices, row):
    toneeds = need_indices
    fromneeds = [row for _ in range(len(toneeds))]
    fullNeedConnectionIndices = (fromneeds, toneeds)
    return fullNeedConnectionIndices

def need_connection_indices(tensor, all_need_indices, test_need_indices):
    allindices = ([],[])
    for row in test_need_indices:
        rowindices = fullrow_need_connection_indices(tensor, all_need_indices, row)
        allindices[0].extend(rowindices[0])
        allindices[1].extend(rowindices[1])
    return allindices

def randomconnection_indices(tensor, numIndices):
    i0 = [random.randint(0,tensor[0].shape[0]-1) for _ in range(numIndices)]
    i1 = [random.randint(0,tensor[0].shape[0]-1) for _ in range(numIndices)]
    return (i0,i1)

def mask_connection(tensor, mask):
    for i in range(len(mask[0])):
        tensor[0][mask[0][i], mask[1][i]] = 0
        tensor[0][mask[1][i], mask[0][i]] = 0
    return tensor

def normalize_predictions(P, mask, k):
    for i in range(len(mask[0])):
        a = mask[0][i]
        b = mask[1][i]
        nrm = norm(P[a, b, :k])
        if nrm != 0:
            P[a, b, :k] = np.round_(P[a, b, :k] / nrm, decimals=3)
    return P

if __name__ == '__main__':

    # load the tensor input data
    K, entities = read_input_tensor(sys.argv[1], sys.argv[2])

    GROUND_TRUTH = K[0].toarray()

    nz = connection_indices(K)
    nz0 = list(set(nz[0]))
    shuffle(nz0)
    nz0 = nz0[0:len(nz0)/10]

    need_indices = need_indices(entities)
    ind = need_connection_indices(K, need_indices, nz0)

    _log.info("number of needs with connections: " + str(len(ind[0]) / len(need_indices)))
    #K = mask_connection(K, nz)

    rank = 100
    _log.info('start processing ...')
    _log.info('Datasize: %d x %d x %d | Rank: %d' % (
        K[0].shape + (len(K),) + (rank,))
    )

    # execute rescal algorithm
    A, R, _, _, _ = rescal_als(
        K, rank, init='nvecs', conv=1e-3,
        lambda_A=0, lambda_R=0, compute_fit='true'
    )
    P = predict_rescal_als(A, R)

    _log.info("start normalizing ...")
    #P = normalize_predictions(P, ind, P.shape[2])


    _log.info("start precision_recall_curve ...")
    prec, recall, thresholds = precision_recall_curve(GROUND_TRUTH[ind], P[:,:,0][ind])

    auc = auc(recall, prec)
    _log.info('AUC: ' + str(auc))


    optimal_threshold = 0.0
    for i in range(len(thresholds)):
        if prec[i] > 0.5:
            optimal_threshold = thresholds[i]
            _log.info('optimal threshold: ' + str(optimal_threshold))
            _log.info('precision: ' + str(prec[i]))
            _log.info('recall: ' + str(recall[i]))
            break

    binary_prediction = [0 if val <= optimal_threshold else 1 for val in P[:,:,0][ind]]
    #prec_score = precision_score(GROUND_TRUTH[ind], binary_prediction)
    #recall_score = recall_score(GROUND_TRUTH[ind], binary_prediction)
    confusionmatrix = confusion_matrix(GROUND_TRUTH[ind], binary_prediction, [1, 0])
    _log.info('confusion matrix: ' + str(confusionmatrix))







