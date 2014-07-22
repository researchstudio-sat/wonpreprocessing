#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Mail Example')

import sys
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
    _log.info("Read mat input file: " + sys.argv[1])
    mat = loadmat(sys.argv[1])

    _log.info("Read header input file: " + sys.argv[2])
    input = open(sys.argv[2])
    entities = input.read().splitlines()

    K = []
    for i in range(0,3):
        G = mat['Rs' + str(i)]
        I,J = G.nonzero()
        V = G.data
        K.append(sparse.csr_matrix((V,(I,J)),G.shape))

    rank = 100
    e = K[0].shape[0]
    k = 1
    SZ = e * e * k;

    _log.info('start processing ...')
    _log.info('Datasize: %d x %d x %d | No. of classes: %d | Rank: %d' % (
        K[0].shape + (len(K),) + (k,) + (rank,))
    )

    # rescal processing
    P = predict_rescal_als(K, rank)

    #_log.info('normalize predictions ...')
    #P = normalize_predictions(P, e, k)

    _log.info('Writing output file: ' + sys.argv[3])
    out = open(sys.argv[3], 'w+')
    for line in range(0, len(entities)):
        darr = np.array(P[line,:,2])
        indices = (np.argsort(darr))[-20:]
        predicted_entities = [entities[i][6:] + " (" + str(round(darr[i], 2)) + ")" for i in reversed(indices)]
        need = entities[line].ljust(150)
        if (need.startswith('Need:')):
            out.write(need + ': ' + ', '.join(predicted_entities) + '\n')

    _log.info('Writing output file: ' + sys.argv[3] + ".con")
    out = open(sys.argv[3] + ".con", 'w+')
    for line in range(0, len(entities)):
        darr = np.array(P[line,:,0])
        indices = (np.argsort(darr))[-20:]
        if darr[indices[-1]] > 0.1:
            newIndices = []
            for i in indices:
                if darr[i] > 0.1:
                    newIndices.append(i)
            indices = newIndices
            predicted_entities = [entities[i] + " (" + str(round(darr[i], 2)) + ")" for i in reversed(indices)]
            need = entities[line].ljust(150)
            if (need.startswith('Need:')):
                out.write(need[6:] + '\n')
                for entity in predicted_entities:
                    if (entity.startswith('Need:')):
                        out.write(entity[6:] + '\n')
            out.write('\n')

