__author__ = 'hfriedrich'

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
_log = logging.getLogger()

import os
import argparse

import numpy as np
from scipy.sparse import lil_matrix
from time import strftime
from tools.tensor_utils import connection_indices, read_input_tensor, SparseTensor
from scripts.evaluation_algorithms import CosineEvaluation, RescalEvaluation, \
    RescalSimilarityEvaluation, PredictionMatrixFileEvaluation, CombineCosineRescalEvaluation, \
    IntersectionCosineRescalEvaluation

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
    parser.add_argument('-rescal', action="store", dest="rescal", nargs=9,
                        metavar=('rank', 'threshold', 'useNeedTypeSlice', 'transitiveConnections', 'init', 'conv',
                                 'lambda_A', 'lambda_R', 'lambda_V'),
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
    parser.add_argument('-prediction_matrix_file', action='store', dest='prediction_matrix_file',
                        help='path to matrix file (same file format as connection slice of tensor) that makes '
                             'connection predictions and can be generated separately')

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

    # by changing this parameter the number of training connections per need can be set. Choose a high value (e.g.
    # 100) to use all connection in the connections file. Choose a low number to restrict the number of training
    # connections (e.g. to 1 or even 0). This way tests are possible that describe situation where initially not many
    # connection are available to learn from.
    MAX_CONNECTIONS_PER_NEED = args.maxconnections

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

    _log.info('Number of test needs: %d (OFFERS: %d, WANTS: %d)' %
              (len(needs), len(set(needs) & set(offers)), len(set(needs) & set(wants))))
    _log.info('Number of total needs: %d (OFFERS: %d, WANTS: %d)' %
              (len(input_tensor.getNeedIndices()), len(offers), len(wants)))
    _log.info('Number of test and train connections: %d' % len(connection_indices(input_tensor)[0]))
    _log.info('Number of total connections (for evaluation): %d' % len(connection_indices(GROUND_TRUTH)[0]))
    _log.info('Number of attributes: %d' % len(input_tensor.getAttributeIndices()))
    _log.info('Starting %d-fold cross validation' % FOLDS)

    # Create the algorithm evaluation classes
    _log.info('Evaluate the following algorithms: ')
    evaluation_algorithms = []
    if args.rescal:
        _log.info('- RescalEvaluation')
        evaluation_algorithms.append(RescalEvaluation(
            args, outfolder, _log, GROUND_TRUTH, start_time))
    if args.rescalsim:
        _log.info('- RescalSimilarityEvaluation')
        evaluation_algorithms.append(RescalSimilarityEvaluation(
            args, outfolder, _log, GROUND_TRUTH, start_time))
    if args.cosine:
        _log.info('- (non-weigthed) CosineEvaluation')
        evaluation_algorithms.append(CosineEvaluation(
            args, outfolder, _log, GROUND_TRUTH, start_time, False))
    if args.cosine_weigthed:
        _log.info('- (weighted) CosineEvaluation')
        evaluation_algorithms.append(CosineEvaluation(
            args, outfolder, _log, GROUND_TRUTH, start_time, True))
    if args.prediction_matrix_file:
        _log.info('- PredictionMatrixFileEvaluation')
        evaluation_algorithms.append(PredictionMatrixFileEvaluation(
            args, outfolder, _log, GROUND_TRUTH, start_time))
    if args.cosine_rescal:
        _log.info('- CombineCosineRescalEvaluation')
        evaluation_algorithms.append(CombineCosineRescalEvaluation(
            args, outfolder, _log, GROUND_TRUTH, start_time))
    if args.intersection:
        _log.info('- IntersectionCosineRescalEvaluation')
        evaluation_algorithms.append(IntersectionCosineRescalEvaluation(
            args, outfolder, _log, GROUND_TRUTH, start_time))

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
        for algorithm in evaluation_algorithms:
            algorithm.evaluate_fold(test_tensor.copy(), test_needs, idx_test)
        # end of fold loop

    # evaluation ended, print the summary
    _log.info('====================================================')
    for algorithm in evaluation_algorithms:
        algorithm.finish_evaluation()
        _log.info('----------------------------------------------------')






