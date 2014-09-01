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

def connection_indices(connection_slice):
    nz = connection_slice.nonzero()
    nz0 = [nz[0][i] for i in range(len(nz[0])) if nz[0][i] <= nz[1][i]]
    nz1 = [nz[1][i] for i in range(len(nz[0])) if nz[0][i] <= nz[1][i]]
    indices = [i for i in range(len(nz0))]
    shuffle(indices)
    ret0 = [nz0[i] for i in indices]
    ret1 = [nz1[i] for i in indices]
    nzsym = (ret0, ret1)
    return nzsym

def need_indices(headers):
    needs = [i for i in range(0, len(headers)) if (headers[i].startswith('Need:'))]
    return needs

def need_connection_indices(all_need_indices, test_need_indices):
    allindices = ([],[])
    for row in test_need_indices[0]:
        fromneeds = [row for _ in range(len(all_need_indices))]
        toneeds = all_need_indices
        allindices[0].extend(fromneeds)
        allindices[1].extend(toneeds)
    return allindices

def mask_connection(tensor, mask):
    Tc = [slice.copy() for slice in tensor]
    for i in range(len(mask[0])):
        Tc[0][mask[0][i], mask[1][i]] = 0
        Tc[0][mask[1][i], mask[0][i]] = 0
    return Tc

def mask_all_connections_of_need(tensor, need):
    Tc = [slice.copy() for slice in tensor]
    for i in range(len(Tc[0])):
            Tc[0][i, need] = 0
            Tc[0][need, i] = 0
    return Tc

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
    input_tensor, entities = read_input_tensor(sys.argv[1], sys.argv[2])
    GROUND_TRUTH = input_tensor[0]

    # 10-fold cross validation
    FOLDS = 10
    offset = 0
    connectionIndices = connection_indices(GROUND_TRUTH)
    need_indices = need_indices(entities)
    foldSize = int(len(connectionIndices[0]) / FOLDS)
    _log.info('Starting %d-fold cross validation' % FOLDS)
    _log.info('Number of connections: %d' % len(connectionIndices[0]))
    _log.info('Fold size: %d' % foldSize)
    for f in range(FOLDS):

        _log.info('Fold %d:' % f)

        # connection indices with entry 1 that get masked by 0
        idx_con_test = (connectionIndices[0][offset:offset+foldSize],
                    connectionIndices[1][offset:offset+foldSize])

        # train connection indices that stay 1
        idx_con_train = (connectionIndices[0][:offset] +
                         connectionIndices[0][offset+foldSize:],
                         connectionIndices[1][:offset] +
                         connectionIndices[1][offset+foldSize:])

        # mask the test connection indices
        test_tensor = mask_connection(input_tensor, idx_con_test)

        # we do not only evaluate the connection indices with entry 1 but all
        # connection indices (to all other needs) of that a need that had a connection index set to 1
        idx_test = need_connection_indices(need_indices, idx_con_test)
        idx_train = need_connection_indices(need_indices, idx_con_train)

        # execute the rescal algorithm
        rank = 100
        _log.info('start rescal processing ...')
        _log.info('Datasize: %d x %d x %d | Rank: %d' % (
            test_tensor[0].shape + (len(test_tensor),) + (rank,))
        )

        A, R, _, _, _ = rescal_als(
            test_tensor, rank, init='nvecs', conv=1e-3,
            lambda_A=0, lambda_R=0, compute_fit='true'
        )
        P = predict_rescal_als(A, R)
        #P = normalize_predictions(P, ind, P.shape[2])

        # evaluate the predictions
        prec, recall, thresholds = precision_recall_curve(GROUND_TRUTH.toarray()[idx_test], P[:,:,0][idx_test])
        areaUnderCurve = auc(recall, prec)
        _log.info('AUC: ' + str(areaUnderCurve))

        optimal_threshold = 0.0
        for i in range(len(thresholds)):
            if prec[i] > 0.5:
                optimal_threshold = thresholds[i]
                _log.info('optimal threshold: ' + str(optimal_threshold))
                _log.info('precision: ' + str(prec[i]))
                _log.info('recall: ' + str(recall[i]))
                break

        binary_prediction = [0 if val <= optimal_threshold else 1 for val in P[:,:,0][idx_test]]

        confusionmatrix = confusion_matrix(GROUND_TRUTH.toarray()[idx_test], binary_prediction, [1, 0])
        _log.info('confusion matrix: ' + str(confusionmatrix))

        offset += foldSize









