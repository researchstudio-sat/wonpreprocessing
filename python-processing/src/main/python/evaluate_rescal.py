#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
_log = logging.getLogger('Mail Example')

import sys
import codecs
import numpy as np
import sklearn.metrics as m
from numpy import dot, zeros
from numpy.random import shuffle
from scipy.io.matlab import loadmat
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy import sparse
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from rescal import rescal_als


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

# read the input tensor data (data-0.mtx ... data-3.mtx) and
# the headers file (headers.txt) from the folder
def read_input_tensor(folder):
    header_file =  folder + "/headers.txt"
    _log.info("Read header input file: " + header_file)
    input = codecs.open(header_file,'r',encoding='utf8')
    headers = input.read().splitlines()
    K = []
    for i in range(0,3):
        data_file = folder + "/data-" + str(i) + ".mtx"
        _log.info("Read the data input file: " + data_file )
        matrix = mmread(data_file)
        K.append(csr_matrix(matrix))
    input.close()
    return K, headers

# return a tuple with two lists holding need indices that represent connections
# between these needs, symmetric connection are only represented once
def connection_indices(tensor):
    nz = tensor[0].nonzero()
    nz0 = [nz[0][i] for i in range(len(nz[0])) if nz[0][i] <= nz[1][i]]
    nz1 = [nz[1][i] for i in range(len(nz[0])) if nz[0][i] <= nz[1][i]]
    indices = [i for i in range(len(nz0))]
    shuffle(indices)
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

# mask all connections but a number of X for each need
def mask_all_but_X_connections_per_need(tensor, keep_x):
    slices = [slice.copy().toarray() for slice in tensor]
    for row in set(tensor[0].nonzero()[0]):
        mask_idx = [col for col in range(len(slices[0][row,:])) if slices[0][row,col] == 1.0]
        shuffle(mask_idx)
        for col in mask_idx[keep_x:]:
            slices[0][row,col] = 0
            slices[0][col,row] = 0
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

# create a similarity matrix of needs (and attributes)
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
def eval_report(y_true, y_pred, f_beta=1):
    precision, recall, fscore, _ = m.precision_recall_fscore_support(y_true, y_pred, f_beta, average='weighted')
    accuracy = m.accuracy_score(y_true, y_pred)
    confusionmatrix = m.confusion_matrix(y_true, y_pred, [1, 0])
    _log.info('accuracy: %f' % accuracy)
    _log.info('precision: %f' % precision)
    _log.info('recall: %f' % recall)
    _log.info('f%d-score: %f' % (f_beta, fscore))
    _log.info('confusion matrix: ' + str(confusionmatrix))
    return precision, recall, accuracy, fscore

# calculate the optimal threshold by maximizing the f-score measure
def get_optimal_threshold(y_true, prediction, f_beta=1):
    prediction = np.round_(prediction, decimals=5)
    prec, recall, thresholds = m.precision_recall_curve(y_true, prediction)
    max_f_score = 0
    optimal_threshold = 0.0
    for i in range(len(thresholds)):
        r = recall[i]
        p = prec[i]
        div = (f_beta * f_beta * p + r)
        if div != 0:
            f_score = (1 + f_beta * f_beta) * (p * r) / div
            if f_score > max_f_score:
                max_f_score = f_score
                optimal_threshold = thresholds[i]
    return optimal_threshold

# class to collect data during the runs of the test and print calculated measures for summary
class EvaluationReport:

    def __init__(self, f_beta=1):
        self.f_beta = f_beta
        self.precision = []
        self.recall = []
        self.accuracy = []
        self.fscore = []

    def add_evaluation_data(self, y_true, y_pred):
        p, r, f, _ =  m.precision_recall_fscore_support(y_true, y_pred, average='weighted')
        a = m.accuracy_score(y_true, y_pred)
        cm = m.confusion_matrix(y_true, y_pred, [1, 0])
        self.precision.append(p)
        self.recall.append(r)
        self.fscore.append(f)
        self.accuracy.append(a)
        _log.info('accuracy: %f' % a)
        _log.info('precision: %f' % p)
        _log.info('recall: %f' % r)
        _log.info('f%d-score: %f' % (self.f_beta, f))
        _log.info('confusion matrix: ' + str(cm))

    def summary(self):
        a = np.array(self.accuracy)
        p = np.array(self.precision)
        r = np.array(self.recall)
        f = np.array(self.fscore)
        _log.info('Accuracy Mean / Std: %f / %f' % (a.mean(), a.std()))
        _log.info('Precision Mean / Std: %f / %f' % (p.mean(), p.std()))
        _log.info('Recall Mean / Std: %f / %f' % (r.mean(), r.std()))
        _log.info('F%d-Score Mean / Std: %f / %f' % (self.f_beta, f.mean(), f.std()))


# This program executes a N-fold cross validation on rescal tensor data.
# For each fold test needs are randomly chosen and all their connections to
# all other needs are masked by 0 in the tensor. Then rescal is executed and
# measures are taken that describe the recovery of these masked connection entries.
# Three different approaches for connection prediction between needs are tested.
# 1) choose a fixed threshold by taking the maximum fscore measure and take
#    every connection that exceeds this threshold
# 2) take the top X highest rated connections as a prediction per need
#    (only match offers with wants)
# 3) take the top X most similar needs to the test need for connection prediction
#    (only match offers with wants)
#
# Input parameters:
# argv[1]: folder with the following files:
# - tensor matrix data file name
# - headers file name
# - connections file name
if __name__ == '__main__':

    # load the tensor input data
    folder = sys.argv[1]
    input_tensor, headers = read_input_tensor(folder)
    checked_needs = manually_checked_needs(headers, folder + "/connections.txt")
    GROUND_TRUTH = [input_tensor[i].copy() for i in range(len(input_tensor))]


    # TEST-PARAMETERS:
    # ===================

    # 10-fold cross validation
    FOLDS = 10

    # changing the rank parameter influences the amount of internal latent "clusters" of the algorithm and thus the
    # quality of the matching as well as performance (memory and execution time)
    RANK = 50

    # the f-beta-measure is used to calculate the optimal threshold for the rescal algorithm. beta=1 is the
    # F1-measure which weights precision and recall both same important. the higher the beta value,
    # the more important is recall compared to precision
    F_BETA = 2

    # this is used to return the top X connections for each need. Used in predict_connections_by_need_similarity and predict_connections_per_need
    TOPX = 10

    # by changing this parameter the number of training connections per need can be set. Choose a high value (e.g.
    # 100) to use all connection in the connections file. Choose a low number to restrict the number of training
    # connections (e.g. to 1 or even 0). This way tests are possible that describe situation where initially not many
    # connection are available to learn from.
    MAX_CONNECTIONS_PER_NEED = 100

    _log.info('------------------------------')
    _log.info('Test Setup:')
    _log.info('------------------------------')
    _log.info('For testing use only needs that were manually checked for connections')
    needs = checked_needs #need_indices(headers)
    _log.info('For testing use a maximum number of %d connections per need' % MAX_CONNECTIONS_PER_NEED)
    input_tensor = mask_all_but_X_connections_per_need(input_tensor, MAX_CONNECTIONS_PER_NEED)
    offers = offer_indices(input_tensor, headers)
    wants = want_indices(input_tensor, headers)
    shuffle(needs)
    fold_size = int(len(needs) / FOLDS)
    AUC_test = zeros(FOLDS)
    report1 = EvaluationReport(F_BETA)
    report2 = EvaluationReport(F_BETA)
    report3 = EvaluationReport(F_BETA)

    _log.info('Number of test needs: %d (OFFERS: %d, WANTS: %d)' %
              (len(needs), len(set(needs) & set(offers)), len(set(needs) & set(wants))))
    _log.info('Number of total needs: %d (OFFERS: %d, WANTS: %d)' %
              (len(need_indices(headers)), len(offers), len(wants)))
    _log.info('Number of test and train connections: %d' % len(connection_indices(input_tensor)[0]))
    _log.info('Number of total connections (for evaluation): %d' % len(connection_indices(GROUND_TRUTH)[0]))
    _log.info('Number of attributes: %d' % (input_tensor[0].shape[0] - len(need_indices(headers))))
    _log.info('Starting %d-fold cross validation' % FOLDS)
    _log.info('Fold size (needs): %d' % fold_size)

    # start the cross validation
    offset = 0
    for f in range(FOLDS):

        _log.info('------------------------------')
        _log.info('Fold %d:' % f)
        _log.info('------------------------------')

        # define test set of needs
        test_needs = needs[offset:offset+fold_size]
        test_tensor = mask_need_connections(input_tensor, test_needs)
        idx_test = need_connection_indices(need_indices(headers), test_needs)

        # execute the rescal algorithm
        P, A, R = predict_rescal_als(test_tensor, RANK)

        # in the first fold-run calculate a threshold to work with
        if f == 0:
            optimal_threshold = get_optimal_threshold(
                np.ravel(GROUND_TRUTH[0].toarray()[idx_test]),
                np.ravel(P[:,:,0][idx_test]), F_BETA)
            _log.info('choose threshold ' + str(optimal_threshold) +
                      ' (maximum F' + str(F_BETA) + '-score of first fold)')

        # evaluate the predictions
        prec, recall, thresholds = m.precision_recall_curve(GROUND_TRUTH[0].toarray()[idx_test], P[:,:,0][idx_test])
        AUC_test[f] = m.auc(recall, prec)
        _log.info('AUC test: ' + str(AUC_test[f]))

        # first use a fixed threshold to compute several measures
        _log.info('For threshold %f (max f%d-score):' % (optimal_threshold, F_BETA))
        P_bin = predict_connections_by_threshold(P, optimal_threshold, offers, wants, test_needs)
        binary_prediction = P_bin[:,:,0][idx_test]
        report1.add_evaluation_data(GROUND_TRUTH[0].toarray()[idx_test], binary_prediction)

        # second use the 10 highest rated connections for every need to other needs
        _log.info('For prediction of %d top rated connections per need: ' % TOPX)
        P_bin = predict_connections_per_need(P, offers, wants, test_needs, TOPX)
        binary_prediction = P_bin[:,:,0][idx_test]
        report2.add_evaluation_data(GROUND_TRUTH[0].toarray()[idx_test], binary_prediction)

        # third use the 10 most similar needs per need to predict connections
        _log.info('For prediction of %d connections based on need similarity: ' % TOPX)
        P_bin = predict_connections_by_need_similarity(A, offers, wants, test_needs, TOPX)
        binary_prediction = P_bin[:,:,0][idx_test]
        report3.add_evaluation_data(GROUND_TRUTH[0].toarray()[idx_test], binary_prediction)

        offset += fold_size

    _log.info('====================================================')
    _log.info('AUC-PR Test Mean / Std: %f / %f' % (AUC_test.mean(), AUC_test.std()))
    _log.info('----------------------------------------------------')
    _log.info('For threshold %f (max f%d-score):' % (optimal_threshold, F_BETA))
    report1.summary()
    _log.info('----------------------------------------------------')
    _log.info('For prediction of %d top rated connections per need: ' % TOPX)
    report2.summary()
    _log.info('----------------------------------------------------')
    _log.info('For prediction of %d connections based on need similarity: ' % TOPX)
    report3.summary()





