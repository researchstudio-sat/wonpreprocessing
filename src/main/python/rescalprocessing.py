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
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def predict_rescal_als(A, R):
    n = A.shape[0]
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return P

def scale_slice(slice, weight):
    scale = csr_matrix([[100.0 for i in range(0, K[0].shape[0])]])
    return slice.multiply(scale);

def similarity_ranking(A):
    dist = squareform(pdist(A, metric='cosine'))
    return dist

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

def write_connection_output(file, input_tensor, predicted_tensor, slice, entities):
    _log.info('Writing output file: ' + file)
    out = open(file, 'w+')
    for line in range(0, len(entities)):
        if (entities[line].startswith('Need:')):
            darr = np.array(predicted_tensor[line,:,slice])
            indices = reversed((np.argsort(darr))[-20:])
            indices = [i for i in indices if darr[i] > 0.1]
            if len(indices) > 0:
                out.write('\n')
                out.write(entities[line][6:] + '\n')
            for i in indices:
                if (entities[i].startswith('Need:')):
                    newPrediction = ("NEW_PREDICTION: " if input_tensor[slice].getrow(line).getcol(i)[0,0] == 0.0 else "")
                    predicted_entities = newPrediction + entities[i] + " (" + str(round(darr[i], 2)) + ")"
                    out.write(predicted_entities + '\n')

def write_need_output(file, similarity_matrix, entities):
    _log.info('Writing output file: ' + file)
    out = open(file, 'w+')
    for line in range(0, len(entities)):
        if (entities[line].startswith('Need:')):
            indices = (np.argsort(similarity_matrix[line,:]))[:10]
            predicted_entities = [entities[i] + " (" + str(round(similarity_matrix[line,i], 4)) + ")" for i in indices if entities[
                i].startswith('Need:') and i != line]
            out.write(entities[line] + '\n' + '\n'.join(predicted_entities) + '\n\n')


if __name__ == '__main__':

    # load the tensor input data
    K, entities = read_input_tensor(sys.argv[1], sys.argv[2])

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

    # topic attributes (terms) are saved in slice 2
    write_term_output(sys.argv[3] + "/outterm.txt", P, 2, entities)

    # connections are saved in slice 0
    write_connection_output(sys.argv[3] + "/outconn.txt", K, P, 0, entities)

    # need similarity - use slices connection and attributes, not classification (slice 1)
    A, R, _, _, _ = rescal_als(
        [K[0],K[2]], rank, init='nvecs', conv=1e-3,
        lambda_A=0, lambda_R=0, compute_fit='true'
    )
    S = similarity_ranking(A)
    write_need_output(sys.argv[3] + "/outneed.txt", S, entities)







