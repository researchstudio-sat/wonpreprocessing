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
                P[a, b, :k] = np.round_(P[a, b, :k] / nrm, decimals=3)
    return P

def read_input_tensor(data_file, header_file):
    _log.info("Read mat input file: " + data_file)
    mat = loadmat(data_file)

    _log.info("Read header input file: " + header_file)
    input = open(header_file)
    headers = input.read().splitlines()

    K = []
    for i in range(0,3):
        G = mat['Rs' + str(i)]
        I,J = G.nonzero()
        V = G.data
        K.append(sparse.csr_matrix((V,(I,J)),G.shape))

    return K, headers

def write_term_output(file, P, slice, entities):
    _log.info('Writing output file: ' + file)
    out = open(file, 'w+')
    for line in range(0, len(entities)):
        darr = np.array(P[line,:,slice])
        indices = (np.argsort(darr))[-20:]
        predicted_entities = [entities[i][6:] + " (" + str(round(darr[i], 2)) + ")" for i in reversed(indices)]
        need = entities[line].ljust(150)
        if (need.startswith('Need:')):
            out.write(need + ': ' + ', '.join(predicted_entities) + '\n')

def write_connection_output(file, P, slice, entities):
    _log.info('Writing output file: ' + file)
    out = open(file, 'w+')
    for line in range(0, len(entities)):
        if (entities[line].startswith('Need:')):
            darr = np.array(P[line,:,slice])
            indices = reversed((np.argsort(darr))[-20:])
            indices = [i for i in indices if darr[i] > 0.1]
            if len(indices) > 0:
                predicted_entities = [entities[i] + " (" + str(round(darr[i], 2)) + ")" for i in indices if entities[
                    i].startswith('Need:')]
                out.write(entities[line][6:] + '\n')
                for entity in predicted_entities:
                    out.write(entity[6:] + '\n')
                out.write('\n')


if __name__ == '__main__':

    # load the tensor input data
    K, entities = read_input_tensor(sys.argv[1], sys.argv[2])

    rank = 100
    _log.info('start processing ...')
    _log.info('Datasize: %d x %d x %d | Rank: %d' % (
        K[0].shape + (len(K),) + (rank,))
    )

    # execute rescal algorithm
    P = predict_rescal_als(K, rank)

    # topic attributes (terms) are saved in slice 2
    write_term_output(sys.argv[3] + "/outterm.txt", P, 2, entities)

    # connections are saved in slice 0
    write_connection_output(sys.argv[3] + "/outconn.txt", P, 0, entities)





