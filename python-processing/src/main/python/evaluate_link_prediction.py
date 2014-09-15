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
import os
from numpy import dot, zeros
from numpy.random import shuffle
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from rescal import rescal_als
from time import strftime
from cosine_link_prediction import cosinus_link_prediciton, cosinus_weighted_link_prediction


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

# classify based on 2 values as true positive (TP), true negative (TN), false positive (FP), false negative (FN)
def test_classification(y_true, y_pred):
    if y_true == y_pred:
        if y_true == 1.0:
            return "TP"
        else:
            return "TN"
    else:
        if y_true == 1.0:
            return "FN"
        else:
            return "FP"

# helper function
def create_file_from_sorted_list(dir, filename, list):
    if not os.path.exists(dir):
        os.makedirs(dir)
    file = codecs.open(dir + "/" + filename,'w+',encoding='utf8')
    list.sort()
    for entry in list:
        file.write(entry + "\n")
    file.close()

def calc_precision(TP, FP):
    return TP / float(TP + FP) if (TP + FP) > 0 else 1.0

def calc_recall(TP, FN):
    return TP / float(TP + FN) if (TP + FN) > 0 else 1.0

def calc_accuracy(TP, TN, FP, FN):
    return (TP + TN) / float(TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 1.0

# in a specified folder create files which represent tested needs. For each of these files print the
# binary classifiers: TP, FP, FN including the (connected/not connected) need names for manual detailed analysis of
# the classification algorithm.
def output_statistic_details(outputpath, headers, con_slice_true, con_slice_pred, idx_test):
    TP, TN, FP, FN = 0,0,0,0
    need_list = []
    need_from = idx_test[0][0]
    need_to = idx_test[1][0]
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    summary_file = codecs.open(outputpath + "/_summary.txt",'a+',encoding='utf8')
    class_label = test_classification(con_slice_true[need_from, need_to], con_slice_pred[need_from, need_to])
    need_list.append(class_label + ": " + headers[need_to])
    for i in range(1,len(idx_test[0])):
        need_from_prev = idx_test[0][i-1]
        need_from = idx_test[0][i]
        need_to = idx_test[1][i]
        if need_from_prev != need_from:
            create_file_from_sorted_list(outputpath, headers[need_from_prev][6:] + ".txt", need_list)
            summary_file.write(headers[need_from_prev][6:])
            summary_file.write(": TP: " + str(TP))
            summary_file.write(": TN: " + str(TN))
            summary_file.write(": FP: " + str(FP))
            summary_file.write(": FN: " + str(FN))
            summary_file.write(": Precision: " + str(calc_precision(TP, FP)))
            summary_file.write(": Recall: " + str(calc_recall(TP, FN)))
            summary_file.write(": Accuracy: " + str(calc_accuracy(TP, TN, FP, FN)) + "\n")
            need_list = []
            TP, TN, FP, FN = 0,0,0,0
        class_label = test_classification(con_slice_true[need_from, need_to], con_slice_pred[need_from, need_to])
        TP += (1 if class_label == "TP" else 0)
        TN += (1 if class_label == "TN" else 0)
        FP += (1 if class_label == "FP" else 0)
        FN += (1 if class_label == "FN" else 0)
        if class_label != "TN":
            need_list.append(class_label + ": " + headers[need_to])
    create_file_from_sorted_list(outputpath, headers[need_from_prev][6:] + ".txt", need_list)
    summary_file.write(headers[need_from_prev][6:])
    summary_file.write(": TP: " + str(TP))
    summary_file.write(": TN: " + str(TN))
    summary_file.write(": FP: " + str(FP))
    summary_file.write(": FN: " + str(FN))
    summary_file.write(": Precision: " + str(calc_precision(TP, FP)))
    summary_file.write(": Recall: " + str(calc_recall(TP, FN)))
    summary_file.write(": Accuracy: " + str(calc_accuracy(TP, TN, FP, FN)) + "\n")
    summary_file.close()

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

# for rescal algorithm output predict the X top rated connections for each of the test_needs
def predict_rescal_connections_per_need(P, all_offers, all_wants, test_needs, num_predictions):
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

# for rescal algorithm output predict connections by fixed threshold for each of the test_needs based on the
# similarity of latent need clusters (higher threshold means higher recall)
def predict_rescal_connections_by_need_similarity(A, threshold, all_offers, all_wants, test_needs):
    S = similarity_ranking(A)
    binary_prediction = zeros(P.shape)
    for need in test_needs:
        if need in all_offers:
            all_needs = all_wants
        elif need in all_wants:
            all_needs = all_offers
        else:
            continue

        for x in all_needs:
            if S[need,x] < threshold:
                binary_prediction[need, x, 0] = 1
    return binary_prediction

# for rescal algorithm output predict connections by fixed threshold (higher threshold means higher precision)
def predict_rescal_connections_by_threshold(P, threshold, all_offers, all_wants, test_needs):
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

# calculate the optimal threshold by maximizing the f-score measure
def get_optimal_threshold(recall, precision, threshold, f_beta=1):
    max_f_score = 0
    optimal_threshold = 0.0
    for i in range(len(threshold)):
        r = recall[i]
        p = precision[i]
        div = (f_beta * f_beta * p + r)
        if div != 0:
            f_score = (1 + f_beta * f_beta) * (p * r) / div
            if f_score > max_f_score:
                max_f_score = f_score
                optimal_threshold = threshold[i]
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
# all other needs are masked by 0 in the tensor. Then link prediction algorithms
# (e.g. RESCAL) are executed and measures are taken that describe the recovery of
# these masked connection entries.
# Five different approaches for connection prediction between needs are tested.
# 1) RESCAL: choose a fixed threshold and take every connection that exceeds this threshold
# 2) RESCAL: take the top X highest rated connections as a prediction per need
# 3) RESCAL: choose a fixed threshold and compare need similarity to predict connections
# 4) compute the cosine similarity between attributes of the needs
# 5) compute the weighted cosine similarity between attributes of the needs
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

    # threshold for RESCAL algorithm connection slice, higher threshold means higher precision
    RESCAL_THRESHOLD = 0.005

    # threshold for RESCAL algorithm need similarity, higher threshold means higher recall
    RESCAL_SIMILARITY_THRESHOLD = 0.08

    # threshold for cosine similarity link prediction algorithm, higher threshold means higher recall
    COSINE_SIMILARITY_THRESHOLD = 0.5

    _log.info('------------------------------')
    _log.info('Test Setup:')
    _log.info('------------------------------')
    _log.info('For testing use only needs that were manually checked for connections')
    needs = checked_needs
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
    report4 = EvaluationReport(F_BETA)
    report5 = EvaluationReport(F_BETA)

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
    start_time = strftime("%Y-%m-%d_%H%M%S")
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

        # evaluate the predictions
        prediction = np.round_(P[:,:,0][idx_test], decimals=5)
        precision, recall, threshold = m.precision_recall_curve(GROUND_TRUTH[0].toarray()[idx_test], prediction)
        optimal_threshold = get_optimal_threshold(recall, precision, threshold, F_BETA)
        _log.info('optimal RESCAL threshold would be ' + str(optimal_threshold) +
                  ' (for maximum F' + str(F_BETA) + '-score)')

        AUC_test[f] = m.auc(recall, precision)
        _log.info('AUC test: ' + str(AUC_test[f]))

        # first use a fixed threshold to compute several measures
        _log.info('For RESCAL prediction with threshold %f:' % RESCAL_THRESHOLD)
        P_bin = predict_rescal_connections_by_threshold(P, RESCAL_THRESHOLD, offers, wants, test_needs)
        binary_pred = P_bin[:,:,0][idx_test]
        report1.add_evaluation_data(GROUND_TRUTH[0].toarray()[idx_test], binary_pred)
        output_statistic_details(folder + "/stats/" + start_time + "/rescal", headers,
                                 GROUND_TRUTH[0].toarray(), P_bin[:,:,0], idx_test)

        # second use the 10 highest rated connections for every need to other needs
        _log.info('For RESCAL prediction of %d top rated connections per need: ' % TOPX)
        P_bin = predict_rescal_connections_per_need(P, offers, wants, test_needs, TOPX)
        binary_pred = P_bin[:,:,0][idx_test]
        report2.add_evaluation_data(GROUND_TRUTH[0].toarray()[idx_test], binary_pred)
        output_statistic_details(folder + "/stats/" + start_time + "/rescal_top10", headers,
                                 GROUND_TRUTH[0].toarray(), P_bin[:,:,0], idx_test)

        # third use the 10 most similar needs per need to predict connections
        _log.info('For RESCAL prediction based on need similarity with threshold: %f' % RESCAL_SIMILARITY_THRESHOLD)
        P_bin = predict_rescal_connections_by_need_similarity(A, RESCAL_SIMILARITY_THRESHOLD, offers, wants, test_needs)
        binary_pred = P_bin[:,:,0][idx_test]
        report3.add_evaluation_data(GROUND_TRUTH[0].toarray()[idx_test], binary_pred)
        output_statistic_details(folder + "/stats/" + start_time + "/rescal_similarity", headers,
                                 GROUND_TRUTH[0].toarray(), P_bin[:,:,0], idx_test)

        # execute the cosine similarity link prediction algorithm
        _log.info('For prediction of cosine similarity between needs with threshold %f:' % COSINE_SIMILARITY_THRESHOLD)
        binary_pred = cosinus_link_prediciton(test_tensor, offers, wants, test_needs, COSINE_SIMILARITY_THRESHOLD)
        report4.add_evaluation_data(GROUND_TRUTH[0].toarray()[idx_test], binary_pred[idx_test])
        output_statistic_details(folder + "/stats/" + start_time + "/cosine", headers,
                                 GROUND_TRUTH[0].toarray(), binary_pred, idx_test)

        # execute the weighted cosine similarity link prediction algorithm
        _log.info('For prediction of weigthed cosine similarity between needs with threshold %f:' % COSINE_SIMILARITY_THRESHOLD)
        binary_pred = cosinus_weighted_link_prediction(test_tensor, offers, wants, test_needs, COSINE_SIMILARITY_THRESHOLD)
        report5.add_evaluation_data(GROUND_TRUTH[0].toarray()[idx_test], binary_pred[idx_test])
        output_statistic_details(folder + "/stats/" + start_time + "/weighted_cosine", headers,
                                 GROUND_TRUTH[0].toarray(), binary_pred, idx_test)

        offset += fold_size

    _log.info('====================================================')
    _log.info('AUC-PR Test Mean / Std: %f / %f' % (AUC_test.mean(), AUC_test.std()))
    _log.info('----------------------------------------------------')
    _log.info('For RESCAL prediction with threshold %f:' % RESCAL_THRESHOLD)
    report1.summary()
    _log.info('----------------------------------------------------')
    _log.info('For RESCAL prediction of %d top rated connections per need: ' % TOPX)
    report2.summary()
    _log.info('----------------------------------------------------')
    _log.info('For RESCAL prediction based on need similarity with threshold: %f' % RESCAL_SIMILARITY_THRESHOLD)
    report3.summary()
    _log.info('----------------------------------------------------')
    _log.info('For prediction of cosine similarity between needs with threshold %f:' % COSINE_SIMILARITY_THRESHOLD)
    report4.summary()
    _log.info('----------------------------------------------------')
    _log.info('For prediction of weigthed cosine similarity between needs with threshold %f:' % COSINE_SIMILARITY_THRESHOLD)
    report5.summary()





