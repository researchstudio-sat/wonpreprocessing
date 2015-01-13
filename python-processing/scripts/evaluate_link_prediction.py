#!/usr/bin/env python

__author__ = 'hfriedrich'

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
_log = logging.getLogger()

import os
import codecs
import argparse

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import sklearn.metrics as m
from time import strftime
from tools.graph_utils import create_gexf_graph
from tools.evaluation_utils import NeedEvaluationDetailDict, NeedEvaluationDetails
from tools.cosine_link_prediction import cosinus_link_prediciton
from tools.tensor_utils import connection_indices, read_input_tensor, \
    predict_rescal_connections_by_need_similarity, predict_rescal_connections_by_threshold, similarity_ranking, \
    matrix_to_array, execute_rescal, predict_rescal_connections_array, SparseTensor, extend_next_hop_transitive_connections

# for all test_needs return all indices (shuffeld) to all other needs in the connection slice
def need_connection_indices(all_needs, test_needs):
    allindices = ([],[])
    for row in test_needs:
        fromneeds = [row] * len(all_needs)
        toneeds = all_needs
        allindices[0].extend(fromneeds)
        allindices[1].extend(toneeds)
    indices = range(len(allindices[0]))
    np.random.shuffle(indices)
    ret0 = [allindices[0][i] for i in indices]
    ret1 = [allindices[1][i] for i in indices]
    return (ret0, ret1)

# mask all connections at specified indices in the tensor
def mask_idx_connections(tensor, indices):
    conSlice = lil_matrix(tensor.getSliceMatrix(SparseTensor.CONNECTION_SLICE))
    for idx in range(len(indices[0])):
        conSlice[indices[0][idx],indices[1][idx]] = 0
        conSlice[indices[1][idx],indices[0][idx]] = 0
    masked_tensor = tensor.copy()
    masked_tensor.addSliceMatrix(conSlice, SparseTensor.CONNECTION_SLICE)
    return masked_tensor

# mask all connections of some needs to all other needs
def mask_need_connections(tensor, needs):
    conSlice = lil_matrix(tensor.getSliceMatrix(SparseTensor.CONNECTION_SLICE))
    for need in needs:
        conSlice[need,:] = lil_matrix(np.zeros(conSlice.shape[0]))
        conSlice[:,need] = lil_matrix(np.zeros(conSlice.shape[0])).transpose()
    masked_tensor = tensor.copy()
    masked_tensor.addSliceMatrix(conSlice, SparseTensor.CONNECTION_SLICE)
    return masked_tensor

# mask all connections but a number of X for each need
def mask_all_but_X_connections_per_need(tensor, keep_x):
    conSlice = lil_matrix(tensor.getSliceMatrix(SparseTensor.CONNECTION_SLICE))
    for row in set(conSlice.nonzero()[0]):
        if conSlice[row,:].getnnz() > keep_x:
            mask_idx = conSlice.nonzero()[1][np.where(conSlice.nonzero()[0]==row)]
            np.random.shuffle(mask_idx)
            for col in mask_idx[keep_x:]:
                conSlice[row,col] = 0
                conSlice[col,row] = 0
    masked_tensor = tensor.copy()
    masked_tensor.addSliceMatrix(conSlice, SparseTensor.CONNECTION_SLICE)
    return masked_tensor

# choose number of x needs to keep and remove all other needs that exceed this number
def keep_x_random_needs(tensor, keep_x):
    rand_needs = tensor.getNeedIndices()
    np.random.shuffle(rand_needs)
    remove_needs = rand_needs[keep_x:]
    return mask_needs(tensor, remove_needs)

# mask all needs with have more than X connections
def mask_needs_with_more_than_X_connections(tensor, x_connections):
    remove_needs = []
    for need in range(tensor.shape[0]):
        if (tensor.getSliceMatrix(SparseTensor.CONNECTION_SLICE)[need,].sum() > x_connections):
            remove_needs.append(need)
    return mask_needs(tensor, remove_needs)

# mask complete needs (including references to attributes, connections, etc)
def mask_needs(tensor, needs):
    if (len(needs) == 0):
        return tensor
    slices = [lil_matrix(slice) for slice in tensor.getSliceMatrixList()]
    idx = 0
    newHeaders = ["NULL" if i in needs else tensor.getHeaders()[i] for i in range(len(tensor.getHeaders()))]
    masked_tensor = SparseTensor(newHeaders)
    for slice in slices:
        for need in needs:
            slice[need,:] = lil_matrix(np.zeros(slice.shape[0]))
            slice[:,need] = lil_matrix(np.zeros(slice.shape[0])).transpose()
        masked_tensor.addSliceMatrix(slice, idx)
        idx += 1
    return masked_tensor

# predict connections by combining the execution of algorithms. First execute the cosine similarity
# algorithm (preferably choosing a threshold to get a high precision) and with this predicted matches execute the
# rescal algorithm afterwards (to increase the recall)
def predict_combine_cosine_rescal(input_tensor, test_needs, idx_test, rank,
                                  rescal_threshold, cosine_threshold, useNeedTypeSlice=False):

    wants = input_tensor.getWantIndices()
    offers = input_tensor.getOfferIndices()

    # execute the cosine algorithm first
    binary_pred_cosine = cosinus_link_prediciton(input_tensor, test_needs, cosine_threshold, 0.0, False)

    # use the connection prediction of the cosine algorithm as input for rescal
    temp_tensor = input_tensor.copy()
    temp_tensor.addSliceMatrix(binary_pred_cosine, SparseTensor.CONNECTION_SLICE)
    A,R = execute_rescal(temp_tensor, rank)
    P_bin = predict_rescal_connections_by_threshold(A, R, rescal_threshold, offers, wants, test_needs)

    # return both predictions the earlier cosine and the combined rescal
    binary_pred_cosine = binary_pred_cosine[idx_test]
    binary_pred_rescal = matrix_to_array(P_bin, idx_test)
    return binary_pred_cosine, binary_pred_rescal

# predict connections by combining the execution of algorithms. Compute the predictions of connections for both
# cosine similarity and rescal algorithm. Then return the intersection of the predictions
def predict_intersect_cosine_rescal(input_tensor, test_needs, idx_test, rank,
                                    rescal_threshold, cosine_threshold, useNeedTypeSlice=False):

    wants = input_tensor.getWantIndices()
    offers = input_tensor.getOfferIndices()

    # execute the cosine algorithm
    binary_pred_cosine = cosinus_link_prediciton(input_tensor, test_needs, cosine_threshold, 0.0, False)

    # execute the rescal algorithm
    A,R = execute_rescal(input_tensor, rank)
    P_bin = predict_rescal_connections_by_threshold(A, R, rescal_threshold, offers, wants, test_needs)

    # return the intersection of the prediction of both algorithms
    binary_pred_cosine = binary_pred_cosine[idx_test]
    binary_pred_rescal = matrix_to_array(P_bin, idx_test)
    binary_pred = [min(binary_pred_cosine[i], binary_pred_rescal[i]) for i in range(len(binary_pred_cosine))]
    return binary_pred, binary_pred_cosine, binary_pred_rescal

# write precision/recall (and threshold) curve to file
def write_precision_recall_curve_file(folder, outfilename, precision, recall, threshold):
    if not os.path.exists(folder):
        os.makedirs(folder)
    _log.info("write precision-recall-curve file:" + folder + "/" + outfilename)
    file = codecs.open(folder + "/" + outfilename,'w+',encoding='utf8')
    file.write("precision, recall, threshold")
    prevline = ""
    for i in range(1, len(threshold)):
        line = "\n%.3f, %.3f, %.3f" % (precision[i], recall[i], threshold[i])
        if line != prevline:
            file.write(line)
            prevline = line
    file.close()

# write ROC curve with TP and FP (and threshold) to file
def write_ROC_curve_file(folder, outfilename, TP, FP, threshold):
    if not os.path.exists(folder):
        os.makedirs(folder)
    _log.info("write ROC-curve file:" + folder + "/" + outfilename)
    file = codecs.open(folder + "/" + outfilename,'w+',encoding='utf8')
    file.write("TP, FP, threshold")
    prevline = ""
    for i in range(1, len(threshold)):
        line = "\n%.3f, %.3f, %.3f" % (TP[i], FP[i], threshold[i])
        if line != prevline:
            file.write(line)
            prevline = line
    file.close()

# helper function
def create_file_from_sorted_list(dir, filename, list):
    if not os.path.exists(dir):
        os.makedirs(dir)
    file = codecs.open(dir + "/" + filename,'w+',encoding='utf8')
    list.sort()
    for entry in list:
        file.write(entry + "\n")
    file.close()


# in a specified folder create files which represent tested needs. For each of these files print the
# binary classifiers: TP, FP, FN including the (connected/not connected) need names for manual detailed analysis of
# the classification algorithm.
def output_statistic_details(outputpath, headers, needEvaluationDetailDict, printThresholds=False):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    summary_file = codecs.open(outputpath + "/_summary.txt",'a+',encoding='utf8')

    for need in needEvaluationDetailDict.dict.keys():
        # write need details file
        needEval = needEvaluationDetailDict.dict[need]
        if printThresholds:
            needDetails = [ "TP (%f): " % needEval.TP_thresholds[i] + headers[needEval.TP_toNeeds[i]][6:]
                            for i in range(len(needEval.TP_toNeeds))]
            needDetails += [ "FN (%f): " % needEval.FN_thresholds[i] + headers[needEval.FN_toNeeds[i]][6:]
                             for i in range(len(needEval.FN_toNeeds))]
            needDetails += [ "FP (%f): " % needEval.FP_thresholds[i] + headers[needEval.FP_toNeeds[i]][6:]
                             for i in range(len(needEval.FP_toNeeds))]
        else:
            needDetails = [ "TP: " + headers[toNeed][6:] for toNeed in needEval.TP_toNeeds]
            needDetails += [ "FN: " + headers[toNeed][6:] for toNeed in needEval.FN_toNeeds]
            needDetails += [ "FP: " + headers[toNeed][6:] for toNeed in needEval.FP_toNeeds]

        create_file_from_sorted_list(outputpath, headers[need][6:] + ".txt", needDetails)

        # write the summary file
        summary_file.write(headers[need][6:])
        summary_file.write(": TP: " + str(needEvaluationDetailDict.dict[need].TP))
        summary_file.write(": TN: " + str(needEvaluationDetailDict.dict[need].TN))
        summary_file.write(": FP: " + str(needEvaluationDetailDict.dict[need].FP))
        summary_file.write(": FN: " + str(needEvaluationDetailDict.dict[need].FN))
        summary_file.write(": Precision: " + str(needEvaluationDetailDict.dict[need].getPrecision()))
        summary_file.write(": Recall: " + str(needEvaluationDetailDict.dict[need].getRecall()))
        summary_file.write(": f%f-score : " % F_BETA + str(needEvaluationDetailDict.dict[need].getFScore(F_BETA)))
        summary_file.write(": Accuracy: " + str(needEvaluationDetailDict.dict[need].getAccuracy()) + "\n")
    summary_file.close()


# calculate the optimal threshold by maximizing the f-score measure
def get_optimal_threshold(recall, precision, threshold, f_beta=1.0):
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

    def __init__(self, f_beta=1.0):
        self.f_beta = f_beta
        self.precision = []
        self.recall = []
        self.accuracy = []
        self.fscore = []

    def add_evaluation_data(self, y_true, y_pred):
        p, r, f, _ =  m.precision_recall_fscore_support(y_true, y_pred, average='weighted', beta=self.f_beta)
        a = m.accuracy_score(y_true, y_pred)
        cm = m.confusion_matrix(y_true, y_pred, [1, 0])
        self.precision.append(p)
        self.recall.append(r)
        self.fscore.append(f)
        self.accuracy.append(a)
        _log.info('accuracy: %f' % a)
        _log.info('precision: %f' % p)
        _log.info('recall: %f' % r)
        _log.info('f%.01f-score: %f' % (self.f_beta, f))
        _log.info('confusion matrix: ' + str(cm))

    def summary(self):
        a = np.array(self.accuracy)
        p = np.array(self.precision)
        r = np.array(self.recall)
        f = np.array(self.fscore)
        _log.info('Accuracy Mean / Std: %f / %f' % (a.mean(), a.std()))
        _log.info('Precision Mean / Std: %f / %f' % (p.mean(), p.std()))
        _log.info('Recall Mean / Std: %f / %f' % (r.mean(), r.std()))
        _log.info('F%.01f-Score Mean / Std: %f / %f' % (self.f_beta, f.mean(), f.std()))


# This program executes a N-fold cross validation on rescal tensor data.
# For each fold test needs are randomly chosen and all their connections to
# all other needs are masked by 0 in the tensor. Then link prediction algorithms
# (e.g. RESCAL) are executed and measures are taken that describe the recovery of
# these masked connection entries.
# Different approaches for connection prediction between needs are tested:
# 1) RESCAL: choose a fixed threshold and take every connection that exceeds this threshold
# 2) RESCALSIM: choose a fixed threshold and compare need similarity to predict connections
# 3) COSINE: compute the cosine similarity between attributes of the needs
# 4) COSINE_WEIGHTED: compute the weighted cosine similarity between attributes of the needs
if __name__ == '__main__':

    # CLI processing
    parser = argparse.ArgumentParser(description='link prediction algorithm evaluation script')

    # general
    parser.add_argument('-inputfolder',
                        action="store", dest="inputfolder", required=True,
                        help="input folder of the evaluation")
    parser.add_argument('-outputfolder',
                        action="store", dest="outputfolder", required=False,
                        help="output folder of the evaluation")
    parser.add_argument('-header',
                        action="store", dest="headers", default="headers.txt",
                        help="name of header file")
    parser.add_argument('-connection_slice',
                        action="store", dest="connection_slice", default="connection.mtx",
                        help="name of connection slice file of the tensor")
    parser.add_argument('-needtype_slice',
                        action="store", dest="needtype_slice", default="needtype.mtx",
                        help="name of needtype slice file of the tensor")
    parser.add_argument('-additional_slices', action="store", required=True,
                        dest="additional_slices", nargs="+",
                        help="name of additional slice files to add to the tensor")

    # evaluation parameters
    parser.add_argument('-folds', action="store", dest="folds", default=10,
                        type=int, help="number of folds in cross fold validation")
    parser.add_argument('-maskrandom', action="store_true", dest="maskrandom",
                        help="mask random test connections (not per need)")
    parser.add_argument('-fbeta', action="store", dest="fbeta", default=0.5,
                        type=float, help="f-beta measure to calculate during evaluation")
    parser.add_argument('-maxconnections', action="store", dest="maxconnections", default=1000,
                        type=int, help="maximum number of connections used to lern from per need")
    parser.add_argument('-numneeds', action="store", dest="numneeds", default=10000,
                        type=int, help="number of needs used for the evaluation")
    parser.add_argument('-statistics', action="store_true", dest="statistics",
                        help="write detailed statistics for the evaluation")
    parser.add_argument('-maxhubsize', action="store", dest="maxhubsize", default=10000,
                        type=int, help="use only needs for the evaluation that do not exceed a number X of connections")

    # algorithm parameters
    parser.add_argument('-rescal', action="store", dest="rescal", nargs=4,
                        metavar=('rank', 'threshold', 'useNeedTypeSlice', 'transitiveConnections'),
                        help="evaluate RESCAL algorithm")
    parser.add_argument('-rescalsim', action="store", dest="rescalsim", nargs=4,
                        metavar=('rank', 'threshold', 'useNeedTypeSlice', 'useConnectionSlice'),
                        help="evaluate RESCAL similarity algorithm")
    parser.add_argument('-cosine', action="store", dest="cosine", nargs=2,
                        metavar=('threshold', 'transitive_threshold'),
                        help="evaluate cosine similarity algorithm" )
    parser.add_argument('-cosine_weighted', action="store", dest="cosine_weigthed",
                        nargs=2, metavar=('threshold', 'transitive_threshold'),
                        help="evaluate weighted cosine similarity algorithm")
    parser.add_argument('-cosine_rescal', action="store", dest="cosine_rescal",
                        nargs=4, metavar=('rescal_rank', 'rescal_threshold', 'cosine_threshold', 'useNeedTypeSlice'),
                        help="evaluate combined algorithms cosine similarity and rescal")
    parser.add_argument('-intersection', action="store", dest="intersection",
                        nargs=4, metavar=('rescal_rank', 'rescal_threshold', 'cosine_threshold', 'useNeedTypeSlice'),
                        help="compute the prediction intersection of algorithms cosine similarity and rescal")

    args = parser.parse_args()
    folder = args.inputfolder

    start_time = strftime("%Y-%m-%d_%H%M%S")
    if args.outputfolder:
        outfolder = args.outputfolder
    else:
        outfolder = folder + "/out/" + start_time
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    hdlr = logging.FileHandler(outfolder + "/eval_result_" + start_time + ".log")
    _log.addHandler(hdlr)

    # load the tensor input data
    data_input = [folder + "/" + args.connection_slice,
                  folder + "/" + args.needtype_slice]
    for slice in args.additional_slices:
        data_input.append(folder + "/" + slice)
    header_input = folder + "/" + args.headers
    slices = SparseTensor.defaultSlices + [SparseTensor.ATTR_CONTENT_SLICE, SparseTensor.CATEGORY_SLICE]
    input_tensor = read_input_tensor(header_input, data_input, slices, True)


    # TEST-PARAMETERS:
    # ===================

    # (10-)fold cross validation
    FOLDS = args.folds

    # True means: for testing mask all connections of random test needs (Test Case: Predict connections for new need
    # without connections)
    # False means: for testing mask random connections (Test Case: Predict connections for existing need which may
    # already have connections)
    MASK_ALL_CONNECTIONS_OF_TEST_NEED = not args.maskrandom

    # the f-beta-measure is used to calculate the optimal threshold for the rescal algorithm. beta=1 is the
    # F1-measure which weights precision and recall both same important. the higher the beta value,
    # the more important is recall compared to precision
    F_BETA = args.fbeta

    # by changing this parameter the number of training connections per need can be set. Choose a high value (e.g.
    # 100) to use all connection in the connections file. Choose a low number to restrict the number of training
    # connections (e.g. to 1 or even 0). This way tests are possible that describe situation where initially not many
    # connection are available to learn from.
    MAX_CONNECTIONS_PER_NEED = args.maxconnections

    # changing the rank parameter influences the amount of internal latent "clusters" of the algorithm and thus the
    # quality of the matching as well as performance (memory and execution time)
    RESCAL_RANK = (int(args.rescal[0]) if args.rescal else None)
    RESCAL_SIMILARITY_RANK = (int(args.rescalsim[0]) if args.rescalsim else None)

    # threshold for RESCAL algorithm connection slice, higher threshold means higher precision
    RESCAL_THRESHOLD = (float(args.rescal[1]) if args.rescal else None)

    # threshold for RESCAL algorithm need similarity, higher threshold means higher recall
    RESCAL_SIMILARITY_THRESHOLD = (float(args.rescalsim[1]) if args.rescalsim else None)

    # thresholds for cosine similarity link prediction algorithm, higher threshold means higher recall.
    # set transitive threshold < threshold to avoid transitive predictions
    COSINE_SIMILARITY_THRESHOLD = (float(args.cosine[0]) if args.cosine else None)
    COSINE_SIMILARITY_TRANSITIVE_THRESHOLD = (float(args.cosine[1]) if args.cosine else None)
    COSINE_WEIGHTED_SIMILARITY_THRESHOLD = (float(args.cosine_weigthed[0]) if args.cosine_weigthed else None)
    COSINE_WEIGHTED_SIMILARITY_TRANSITIVE_THRESHOLD = (float(args.cosine_weigthed[1]) if args.cosine_weigthed else None)

    _log.info('------------------------------')
    _log.info('Test Setup:')
    _log.info('------------------------------')


    if (args.numneeds < len(input_tensor.getNeedIndices())):
        input_tensor = keep_x_random_needs(input_tensor, args.numneeds)

    input_tensor = mask_needs_with_more_than_X_connections(input_tensor, args.maxhubsize)
    _log.info('Use only needs that do not have more than %d connections' % args.maxhubsize)

    GROUND_TRUTH = input_tensor.copy()
    needs = input_tensor.getNeedIndices()
    np.random.shuffle(needs)
    connections = need_connection_indices(input_tensor.getNeedIndices(), needs)

    if MASK_ALL_CONNECTIONS_OF_TEST_NEED:
        _log.info('Mask all connections of random test needs (Test Case: Predict connections for new need '
                  'without connections)')
    else:
        _log.info('Mask random connections (Test Case: Predict connections for existing need which may '
                  'already have connections)')

    _log.info('Use a maximum number of %d connections per need' % MAX_CONNECTIONS_PER_NEED)
    input_tensor = mask_all_but_X_connections_per_need(input_tensor, MAX_CONNECTIONS_PER_NEED)
    offers = input_tensor.getOfferIndices()
    wants = input_tensor.getWantIndices()
    need_fold_size = int(len(needs) / FOLDS)
    connection_fold_size = int(len(connections[0]) / FOLDS)
    AUC_test = np.zeros(FOLDS)
    report = [EvaluationReport(F_BETA) for _ in range(9)]
    evalDetails = [NeedEvaluationDetailDict() for _ in range(4)]

    _log.info('Number of test needs: %d (OFFERS: %d, WANTS: %d)' %
              (len(needs), len(set(needs) & set(offers)), len(set(needs) & set(wants))))
    _log.info('Number of total needs: %d (OFFERS: %d, WANTS: %d)' %
              (len(input_tensor.getNeedIndices()), len(offers), len(wants)))
    _log.info('Number of test and train connections: %d' % len(connection_indices(input_tensor)[0]))
    _log.info('Number of total connections (for evaluation): %d' % len(connection_indices(GROUND_TRUTH)[0]))
    _log.info('Number of attributes: %d' % len(input_tensor.getAttributeIndices()))

    _log.info('Starting %d-fold cross validation' % FOLDS)

    # start the cross validation
    offset = 0
    for f in range(FOLDS):

        _log.info('------------------------------')
        # define test set of connections indices
        if MASK_ALL_CONNECTIONS_OF_TEST_NEED:
            # choose the test needs for the fold and mask all connections of them to other needs
            _log.info('Fold %d, fold size %d needs (out of %d)' % (f, need_fold_size, len(needs)))
            test_needs = needs[offset:offset+need_fold_size]
            test_tensor = mask_need_connections(input_tensor, test_needs)
            idx_test = need_connection_indices(input_tensor.getNeedIndices(), test_needs)
            offset += need_fold_size
        else:
            # choose test connections to mask independently of needs
            _log.info('Fold %d, fold size %d connection indices (out of %d)' % (f, connection_fold_size,
                                                                                len(connections[0])))
            idx_test = (connections[0][offset:offset+connection_fold_size],
                        connections[1][offset:offset+connection_fold_size])
            test_tensor = mask_idx_connections(input_tensor, idx_test)

            offset += connection_fold_size
            test_needs = needs
        _log.info('------------------------------')

        # evaluate the algorithms
        if args.rescal:

            # set transitive connections before execution
            if (args.rescal[3] == 'True'):
                _log.info('extend connections transitively to the next need for RESCAL learning')
                test_tensor = extend_next_hop_transitive_connections(test_tensor)

            # execute the rescal algorithm
            useNeedTypeSlice = (args.rescal[2] == 'True')
            A, R = execute_rescal(test_tensor, RESCAL_RANK, useNeedTypeSlice)

            # evaluate the predictions
            _log.info('start predict connections ...')
            prediction = np.round_(predict_rescal_connections_array(A, R, idx_test), decimals=5)
            _log.info('stop predict connections')
            precision, recall, threshold = m.precision_recall_curve(
                GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE, idx_test), prediction)
            optimal_threshold = get_optimal_threshold(recall, precision, threshold, F_BETA)
            _log.info('optimal RESCAL threshold would be ' + str(optimal_threshold) +
                      ' (for maximum F' + str(F_BETA) + '-score)')

            AUC_test[f] = m.auc(recall, precision)
            _log.info('AUC test: ' + str(AUC_test[f]))

            # use a fixed threshold to compute several measures
            _log.info('For RESCAL prediction with threshold %f:' % RESCAL_THRESHOLD)
            P_bin = predict_rescal_connections_by_threshold(A, R, RESCAL_THRESHOLD, offers, wants, test_needs)
            binary_pred = matrix_to_array(P_bin, idx_test)
            report[0].add_evaluation_data(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE,
                                                                              idx_test), binary_pred)
            if args.statistics:
                write_precision_recall_curve_file(outfolder + "/statistics/rescal_" + start_time,
                                                  "precision_recall_curve_fold%d.csv" % f, precision, recall, threshold)
                TP, FP, threshold = m.roc_curve(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE, idx_test), prediction)
                write_ROC_curve_file(outfolder + "/statistics/rescal_" + start_time, "ROC_curve_fold%d.csv" % f, TP, FP, threshold)
                evalDetails[0].add_statistic_details(GROUND_TRUTH.getSliceMatrix(SparseTensor.CONNECTION_SLICE),
                                                     P_bin, idx_test, prediction)

        if args.rescalsim:
            # execute the rescal algorithm
            useNeedTypeSlice = (args.rescalsim[2] == 'True')
            useConnectionSlice = (args.rescalsim[3] == 'True')
            A, R = execute_rescal(test_tensor, RESCAL_SIMILARITY_RANK, useNeedTypeSlice, useConnectionSlice)

            # use the most similar needs per need to predict connections
            _log.info('For RESCAL prediction based on need similarity with threshold: %f' % RESCAL_SIMILARITY_THRESHOLD)
            P_bin = predict_rescal_connections_by_need_similarity(A, RESCAL_SIMILARITY_THRESHOLD, offers, wants, test_needs)
            binary_pred = matrix_to_array(P_bin, idx_test)
            report[1].add_evaluation_data(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE,
                                                                              idx_test), binary_pred)

            if args.statistics:
                S = similarity_ranking(A)
                y_prop = [1.0 - i for i in np.nan_to_num(S[idx_test])]
                precision, recall, threshold = m.precision_recall_curve(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE, idx_test), y_prop)
                write_precision_recall_curve_file(outfolder + "/statistics/rescalsim_" + start_time, "precision_recall_curve_fold%d.csv" % f, precision, recall, threshold)
                TP, FP, threshold = m.roc_curve(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE, idx_test), y_prop)
                write_ROC_curve_file(outfolder + "/statistics/rescalsim_" + start_time, "ROC_curve_fold%d.csv" % f, TP, FP, threshold)
                evalDetails[1].add_statistic_details(GROUND_TRUTH.getSliceMatrix(SparseTensor.CONNECTION_SLICE),
                                                     P_bin, idx_test)

        if args.cosine:
            # execute the cosine similarity link prediction algorithm
            _log.info('For prediction of cosine similarity between needs with thresholds: %f, %f'
                      ':' % (COSINE_SIMILARITY_THRESHOLD, COSINE_SIMILARITY_TRANSITIVE_THRESHOLD))
            binary_pred = cosinus_link_prediciton(test_tensor, test_needs, COSINE_SIMILARITY_THRESHOLD,
                                                  COSINE_SIMILARITY_TRANSITIVE_THRESHOLD, False)
            report[2].add_evaluation_data(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE,idx_test),
                                          matrix_to_array(binary_pred, idx_test))
            if args.statistics:
                evalDetails[2].add_statistic_details(GROUND_TRUTH.getSliceMatrix(SparseTensor.CONNECTION_SLICE),
                                                     binary_pred, idx_test)

        if args.cosine_weigthed:
            # execute the weighted cosine similarity link prediction algorithm
            _log.info('For prediction of weigthed cosine similarity between needs with thresholds %f, %f:' %
                      (COSINE_WEIGHTED_SIMILARITY_THRESHOLD, COSINE_WEIGHTED_SIMILARITY_TRANSITIVE_THRESHOLD))
            binary_pred = cosinus_link_prediciton(test_tensor, test_needs, COSINE_WEIGHTED_SIMILARITY_THRESHOLD,
                                                  COSINE_WEIGHTED_SIMILARITY_TRANSITIVE_THRESHOLD, True)
            report[3].add_evaluation_data(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE, idx_test),
                                          matrix_to_array(binary_pred, idx_test))
            if args.statistics:
                evalDetails[3].add_statistic_details(GROUND_TRUTH.getSliceMatrix(SparseTensor.CONNECTION_SLICE),
                                                     binary_pred, idx_test)

        if args.cosine_rescal:
            cosine_pred, rescal_pred = predict_combine_cosine_rescal(test_tensor, test_needs, idx_test,
                                                                     int(args.cosine_rescal[0]),
                                                                     float(args.cosine_rescal[1]),
                                                                     float(args.cosine_rescal[2]),
                                                                     bool(args.cosine_rescal[3]))
            _log.info('First step for prediction of cosine similarity with threshold: %f:' % float(args.cosine_rescal[2]))
            report[4].add_evaluation_data(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE,
                                                                              idx_test), cosine_pred)
            _log.info('And second step for combined RESCAL prediction with parameters: %d, %f:'
                      % (int(args.cosine_rescal[0]), float(args.cosine_rescal[1])))
            report[5].add_evaluation_data(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE,
                                                                              idx_test), rescal_pred)

        if args.intersection:
            inter_pred, cosine_pred, rescal_pred = predict_intersect_cosine_rescal(test_tensor, test_needs, idx_test,
                                                                                   int(args.intersection[0]), float(args.intersection[1]),
                                                                                   float(args.intersection[2]), bool(args.intersection[3]))
            _log.info('Intersection of predictions of cosine similarity and rescal algorithms: ')
            report[8].add_evaluation_data(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE,
                                                                              idx_test), inter_pred)

            _log.info('For RESCAL prediction with threshold %f:' % float(args.intersection[1]))
            report[7].add_evaluation_data(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE,
                                                                             idx_test), rescal_pred)

            _log.info('For prediction of cosine similarity between needs with thresholds: %f:' %
                      float(args.intersection[2]))
            report[6].add_evaluation_data(GROUND_TRUTH.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE,
                                                                             idx_test), cosine_pred)

        # end of fold loop

    _log.info('====================================================')
    if args.rescal:
        _log.info('AUC-PR Test Mean / Std: %f / %f' % (AUC_test.mean(), AUC_test.std()))
        _log.info('----------------------------------------------------')
        _log.info('For RESCAL prediction with threshold %f:' % RESCAL_THRESHOLD)
        report[0].summary()
        if args.statistics:
            output_statistic_details(outfolder + "/statistics/rescal_" + start_time, GROUND_TRUTH.getHeaders(),
                                     evalDetails[0], True)
            gexf = create_gexf_graph(input_tensor, evalDetails[0])
            output_file = open(outfolder + "/statistics/rescal_" + start_time + "/graph.gexf", "w")
            gexf.write(output_file)
            output_file.close()
        _log.info('----------------------------------------------------')
    if args.rescalsim:
        _log.info('For RESCAL prediction based on need similarity with threshold: %f' % RESCAL_SIMILARITY_THRESHOLD)
        report[1].summary()
        if args.statistics:
            output_statistic_details(outfolder + "/statistics/rescalsim_" + start_time, GROUND_TRUTH.getHeaders(),
                                     evalDetails[1])
            gexf = create_gexf_graph(input_tensor, evalDetails[1])
            output_file = open(outfolder + "/statistics/rescalsim_" + start_time + "/graph.gexf", "w")
            gexf.write(output_file)
            output_file.close()
        _log.info('----------------------------------------------------')
    if args.cosine:
        _log.info('For prediction of cosine similarity between needs with thresholds: %f, %f'
                  ':' % (COSINE_SIMILARITY_THRESHOLD, COSINE_SIMILARITY_TRANSITIVE_THRESHOLD))
        report[2].summary()
        if args.statistics:
            output_statistic_details(outfolder + "/statistics/cosine_" + start_time, GROUND_TRUTH.getHeaders(),
                                     evalDetails[2])
            gexf = create_gexf_graph(input_tensor, evalDetails[2])
            output_file = open(outfolder + "/statistics/cosine_" + start_time + "/graph.gexf", "w")
            gexf.write(output_file)
            output_file.close()
        _log.info('----------------------------------------------------')
    if args.cosine_weigthed:
        _log.info('For prediction of weighted cosine similarity between needs with thresholds: %f, %f'
                  ':' % (COSINE_SIMILARITY_THRESHOLD, COSINE_SIMILARITY_TRANSITIVE_THRESHOLD))
        report[3].summary()
        if args.statistics:
            output_statistic_details(outfolder + "/statistics/wcosine_" + start_time, GROUND_TRUTH.getHeaders(),
                                     evalDetails[3])
            gexf = create_gexf_graph(input_tensor, evalDetails[3])
            output_file = open(outfolder + "/statistics/wcosine_" + start_time + "/graph.gexf", "w")
            gexf.write(output_file)
            output_file.close()
    if args.cosine_rescal:
        _log.info('First step for prediction of cosine similarity with threshold: %f:' % float(args.cosine_rescal[2]))
        report[4].summary()
        _log.info('And second step for combined RESCAL prediction with threshold: %f:' % float(args.cosine_rescal[1]))
        report[5].summary()
    if args.intersection:
        _log.info('Intersection of predictions of cosine similarity and rescal algorithms: ')
        report[8].summary()
        _log.info('For RESCAL prediction with threshold %f:' % float(args.intersection[1]))
        report[7].summary()
        _log.info('For prediction of cosine similarity between needs with thresholds: %f:' %
                  float(args.intersection[2]))
        report[6].summary()





