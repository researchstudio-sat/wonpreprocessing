#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
_log = logging.getLogger('Mail Example')

import sys
import numpy as np
from numpy import dot, zeros
from scipy.io.matlab import loadmat
from scipy.sparse import csr_matrix
from scipy import sparse
from rescal import rescal_als
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import codecs
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, confusion_matrix, accuracy_score
from numpy.random import shuffle

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
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return P, A, R

# read the input tensor data and the headers file
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

# return a list of indices which refer to rows/columns of needs in the tensor
def need_indices(headers):
    needs = [i for i in range(0, len(headers)) if (headers[i].startswith('Need:'))]
    return needs

# return a list of indices which refer to rows/columns of needs of type OFFER in the tensor
def offer_indices(tensor, headers):
    needs = need_indices(headers)
    offer_attr_idx = headers.index("Attr: OFFER")
    offers = [need for need in needs if (tensor[1][need, offer_attr_idx] == 1)]
    return offers

# return a list of indices which refer to rows/columns of needs of type WANT in the tensor
def want_indices(tensor, headers):
    needs = need_indices(headers)
    want_attr_idx = headers.index("Attr: WANT")
    wants = [need for need in needs if (tensor[1][need, want_attr_idx] == 1)]
    return wants

# for all test_needs return all indices to all other needs in the connection slice
def need_connection_indices(all_needs, test_needs):
    allindices = ([],[])
    for row in test_needs:
        fromneeds = [row for _ in range(len(all_needs))]
        toneeds = all_needs
        allindices[0].extend(fromneeds)
        allindices[1].extend(toneeds)
    return allindices

# mask all connections of some needs to all other needs
def mask_need_connections(tensor, needs):
    slices = [slice.copy().toarray() for slice in tensor]
    for need in needs:
        slices[0][need,:] = zeros(tensor[0].shape[0])
        slices[0][:,need] = zeros(tensor[0].shape[0])
    Tc = [csr_matrix(slice) for slice in slices]
    return Tc

# predict the X top rated connections for each of the test_needs
def predict_connections_per_need(P, all_offers, all_wants, test_needs, num_predictions):
    binary_prediction = zeros(P.shape)
    for need in test_needs:
        if need in all_offers:
            all_needs = all_wants
        elif need in all_wants:
            all_needs = all_offers
        else:
            continue
        darr = np.array(P[need,:,0][all_needs])
        pred_indices = (np.argsort(darr))[-num_predictions:]
        for x in np.array(all_needs)[pred_indices]:
            binary_prediction[need, x, 0] = 1
    return binary_prediction

def similarity_ranking(A):
    dist = squareform(pdist(A, metric='cosine'))
    return dist

# predict the X top rated connections for each of the test_needs based on the similarity of latent need clusters
def predict_connections_by_need_similarity(A, all_offers, all_wants, test_needs, num_predictions):
    S = similarity_ranking(A)
    binary_prediction = zeros(P.shape)
    for need in test_needs:
        if need in all_offers:
            all_needs = all_wants
        elif need in all_wants:
            all_needs = all_offers
        else:
            continue
        darr = np.array(S[need,:][all_needs])
        pred_indices = (np.argsort(darr))[:num_predictions]
        for x in np.array(all_needs)[pred_indices]:
            binary_prediction[need, x, 0] = 1
    return binary_prediction

# predict connection by fixed threshold
def predict_connections_by_threshold(P, threshold, all_offers, all_wants, test_needs):
    binary_prediction = zeros(P.shape)
    for need in test_needs:
        if need in all_offers:
            all_needs = all_wants
        elif need in all_wants:
            all_needs = all_offers
        else:
            continue
        for x in all_needs:
            if P[need,x,0] >= threshold:
                binary_prediction[need,x,0] = 1
    return binary_prediction

# calculate several measures based on ground truth and prediction values
def eval_report(y_true, y_pred):
    precision = precision_score(GROUND_TRUTH.toarray()[idx_test], binary_prediction)
    recall = recall_score(GROUND_TRUTH.toarray()[idx_test], binary_prediction)
    accuracy = accuracy_score(GROUND_TRUTH.toarray()[idx_test], binary_prediction)
    confusionmatrix = confusion_matrix(GROUND_TRUTH.toarray()[idx_test], binary_prediction, [1, 0])
    _log.info('accuracy: ' + str(accuracy))
    _log.info('precision: ' + str(precision))
    _log.info('recall: ' + str(recall))
    _log.info('confusion matrix: ' + str(confusionmatrix))
    return precision, recall, accuracy


if __name__ == '__main__':

    # load the tensor input data
    input_tensor, headers = read_input_tensor(sys.argv[1], sys.argv[2])
    GROUND_TRUTH = input_tensor[0]

    # first compute a threshold to work with
    TARGET_PRECISION = 0.1
    optimal_threshold = 0.0
    P,_,_ = predict_rescal_als(input_tensor, 100)
    prec, recall, thresholds = precision_recall_curve(np.ravel(GROUND_TRUTH.toarray()), np.ravel(P[:,:,0]))
    for i in range(len(thresholds)):
        if prec[i] > TARGET_PRECISION:
            optimal_threshold = thresholds[i]
            break
    _log.info('choose threshold ' + str(optimal_threshold) + ' for target precision ' + str(TARGET_PRECISION))

    # 10-fold cross validation
    FOLDS = 10
    offset = 0
    needs = need_indices(headers)
    offers = offer_indices(input_tensor, headers)
    wants = want_indices(input_tensor, headers)
    shuffle(needs)
    fold_size = int(len(needs) / FOLDS)

    AUC_test = zeros(FOLDS)
    precision_threshold = zeros(FOLDS)
    recall_threshold = zeros(FOLDS)
    accuracy_threshold = zeros(FOLDS)
    topX = 10
    precision_topX = zeros(FOLDS)
    recall_topX = zeros(FOLDS)
    accuracy_topX = zeros(FOLDS)
    precision_sim = zeros(FOLDS)
    recall_sim = zeros(FOLDS)
    accuracy_sim = zeros(FOLDS)

    _log.info('Starting %d-fold cross validation' % FOLDS)
    _log.info('Number of needs: %d' % len(needs))
    _log.info('Fold size: %d' % fold_size)
    for f in range(FOLDS):

        _log.info('Fold %d:' % f)

        # define test and training set of needs
        test_needs = needs[offset:offset+fold_size]
        test_tensor = mask_need_connections(input_tensor, test_needs)
        idx_test = need_connection_indices(needs, test_needs)

        # execute the rescal algorithm
        P, A, R = predict_rescal_als(test_tensor, 100)

        # evaluate the predictions
        prec, recall, thresholds = precision_recall_curve(GROUND_TRUTH.toarray()[idx_test], P[:,:,0][idx_test])
        AUC_test[f] = auc(recall, prec)
        _log.info('AUC test: ' + str(AUC_test[f]))

        # first use a fixed threshold to compute several measures
        _log.info('For threshold: ' + str(optimal_threshold))
        P_bin = predict_connections_by_threshold(P, optimal_threshold, offers, wants, test_needs)
        binary_prediction = P_bin[:,:,0][idx_test]
        precision_threshold[f], recall_threshold[f], accuracy_threshold[f] = \
            eval_report(GROUND_TRUTH.toarray()[idx_test], binary_prediction)

        # second use the 10 highest rated connections for every need to other needs
        _log.info('For prediction of %d top rated connections per need: ' % topX)
        P_bin = predict_connections_per_need(P, offers, wants, test_needs, topX)
        binary_prediction = P_bin[:,:,0][idx_test]
        precision_topX[f], recall_topX[f], accuracy_topX[f] = \
            eval_report(GROUND_TRUTH.toarray()[idx_test], binary_prediction)

        # third use the 10 most similar needs per need to predict connections
        _log.info('For prediction of %d top rated connections based on need similarity: ' % topX)
        P_bin = predict_connections_by_need_similarity(A, offers, wants, test_needs, topX)
        binary_prediction = P_bin[:,:,0][idx_test]
        precision_sim[f], recall_sim[f], accuracy_sim[f] = \
            eval_report(GROUND_TRUTH.toarray()[idx_test], binary_prediction)

        offset += fold_size

    _log.info('====================================================')
    _log.info('AUC-PR Test Mean / Std: %f / %f' % (AUC_test.mean(), AUC_test.std()))
    _log.info('----------------------------------------------------')
    _log.info('For threshold: ' + str(optimal_threshold))
    _log.info('Accuracy Mean / Std: %f / %f' % (accuracy_threshold.mean(), accuracy_threshold.std()))
    _log.info('Precision Mean / Std: %f / %f' % (precision_threshold.mean(), precision_threshold.std()))
    _log.info('Recall Mean / Std: %f / %f' % (recall_threshold.mean(), recall_threshold.std()))
    _log.info('----------------------------------------------------')
    _log.info('For prediction of %d top rated connections per need: ' % topX)
    _log.info('Accuracy Mean / Std: %f / %f' % (accuracy_topX.mean(), accuracy_topX.std()))
    _log.info('Precision Mean / Std: %f / %f' % (precision_topX.mean(), precision_topX.std()))
    _log.info('Recall Mean / Std: %f / %f' % (recall_topX.mean(), recall_topX.std()))
    _log.info('----------------------------------------------------')
    _log.info('For prediction of %d top rated connections based on need similarity: ' % topX)
    _log.info('Accuracy Mean / Std: %f / %f' % (accuracy_sim.mean(), accuracy_sim.std()))
    _log.info('Precision Mean / Std: %f / %f' % (precision_sim.mean(), precision_sim.std()))
    _log.info('Recall Mean / Std: %f / %f' % (recall_sim.mean(), recall_sim.std()))






