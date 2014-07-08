#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Mail Example')

import numpy as np
from numpy import dot, array, zeros, setdiff1d
from numpy.linalg import norm
from numpy.random import shuffle
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix, csr_matrix
from scipy import sparse
from sklearn.metrics import precision_recall_curve, auc
from rescal import rescal_als
from sklearn import datasets


def predict_rescal_als(T, rank):
    A, R, _, _, _ = rescal_als(
        T, rank, init='nvecs', conv=1e-3,
        lambda_A=0, lambda_R=0, compute_fit='true'
    )
    n = A.shape[0]
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return P


def normalize_predictions(P, e, k):
    for a in range(e):
        for b in range(e):
            nrm = norm(P[a, b, :k])
            if nrm != 0:
                # round values for faster computation of AUC-PR
                P[a, b, :k] = np.round_(P[a, b, :k] / nrm, decimals=3)
    return P


if __name__ == '__main__':

    # load data
    mat = loadmat('C:/dev/temp/testcorpus/out/rescal/tensordata.mat')
    K = array(mat['Rs'], np.float32)
    rank = 100

    e, k = K.shape[0], K.shape[2]
    SZ = e * e * k;

    # construct array for rescal
    T = [lil_matrix(K[:, :, i]) for i in range(k)]

    _log.info('start processing ...')
    _log.info('Datasize: %d x %d x %d | No. of classes: %d | Rank: %d' % (
        T[0].shape + (len(T),) + (k,) + (rank,))
    )

    # rescal processing
    P = predict_rescal_als(T, rank)

    #_log.info('normalize predictions ...')
    #P = normalize_predictions(P, e, k)

    headersFile = 'C:/dev/temp/testcorpus/out/rescal/headers.txt'
    input = open(headersFile)
    entities = input.read().splitlines()

    _log.info('writing output file  ...')
    outputFile = 'out.txt'
    out = open(outputFile, 'w+')
    for line in range(0, e-1):
        darr = np.array(P[line,:,0])
        indices = (np.argsort(darr))[-20:]
        predicted_entities = [entities[i][6:] + " (" + str(round(darr[i], 2)) + ")" for i in reversed(indices)]
        entities[line] = entities[line].ljust(150)
        out.write(entities[line] + ': ' + ', '.join(predicted_entities) + '\n')



