import numpy as np
import sklearn.metrics as m
from tools.cosine_link_prediction import cosinus_link_prediciton
from tools.evaluation_utils import EvaluationReport, NeedEvaluationDetailDict, get_optimal_threshold, \
    write_ROC_curve_file, write_precision_recall_curve_file
from tools.graph_utils import create_gexf_graph
from tools.tensor_utils import SparseTensor, matrix_to_array, execute_rescal, predict_rescal_connections_by_threshold, \
    read_input_tensor, extend_next_hop_transitive_connections, predict_rescal_connections_array, \
    predict_rescal_connections_by_need_similarity, similarity_ranking

__author__ = 'hfriedrich'

# ========================================================================================
# Abstract class that serves as a base class for the implementation of the evaluation
# of one algorithm during one cross fold validation.
# ========================================================================================
class EvaluationAlgorithm:

    def __init__(self, args, output_folder, logger, input_tensor, start_time):
        self.init(args, output_folder, logger, input_tensor, start_time)

    def init(self, args, output_folder, logger, input_tensor, start_time):
        self.args = args
        self.logger = logger
        self.output_folder = output_folder
        self.report = EvaluationReport(logger, args.fbeta)
        self.ground_truth = input_tensor.copy()
        self.start_time = start_time

    # call this method in the loop at each fold
    def evaluate_fold(self, test_tensor, test_needs, idx_test):
        raise NotImplementedError("not implemented")

    # call this method at the end of the evaluation
    def finish_evaluation(self):
        raise NotImplementedError("not implemented")

# ========================================================================================
# Implementation of evaluation of RESCAL algorithm
# ========================================================================================
# Notes:
# - changing the rank parameter influences the amount of internal latent "clusters" of the
# algorithm and thus the quality of the matching as well as performance (memory and
# execution time).
# - higher threshold for RESCAL algorithm need similarity means higher recall
# ========================================================================================
class RescalEvaluation(EvaluationAlgorithm):

    def __init__(self, args, output_folder, logger, ground_truth, start_time):
        self.init(args, output_folder, logger, ground_truth, start_time)
        self.rank = int(args.rescal[0])
        self.threshold = float(args.rescal[1])
        self.evalDetails = NeedEvaluationDetailDict()
        self.AUC_test = []
        self.foldNumber = 0
        self.offers = ground_truth.getOfferIndices()
        self.wants = ground_truth.getWantIndices()

    def log1(self):
        self.logger.info('For RESCAL prediction with threshold %f:' % self.threshold)

    def evaluate_fold(self, test_tensor, test_needs, idx_test):
        # set transitive connections before execution
        if (self.args.rescal[3] == 'True'):
            self.logger.info('extend connections transitively to the next need for RESCAL learning')
            test_tensor = extend_next_hop_transitive_connections(test_tensor)

        # execute the rescal algorithm
        useNeedTypeSlice = (self.args.rescal[2] == 'True')
        A, R = execute_rescal(
            test_tensor, self.rank, useNeedTypeSlice, init=self.args.rescal[4],
            conv=float(self.args.rescal[5]), lambda_A=float(self.args.rescal[6]),
            lambda_R=float(self.args.rescal[7]), lambda_V=float(self.args.rescal[8]))

        # evaluate the predictions
        self.logger.info('start predict connections ...')
        prediction = np.round_(predict_rescal_connections_array(A, R, idx_test), decimals=5)
        self.logger.info('stop predict connections')
        precision, recall, threshold = m.precision_recall_curve(
            self.ground_truth.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE, idx_test), prediction)
        optimal_threshold = get_optimal_threshold(recall, precision, threshold, self.args.fbeta)
        self.logger.info('optimal RESCAL threshold would be ' + str(optimal_threshold) +
                  ' (for maximum F' + str(self.args.fbeta) + '-score)')
        auc = m.auc(recall, precision)
        self.AUC_test.append(auc)
        self.logger.info('AUC test: ' + str(auc))

        # use a fixed threshold to compute several measures
        self.log1()
        P_bin = predict_rescal_connections_by_threshold(A, R, self.threshold, self.offers, self.wants, test_needs)
        binary_pred = matrix_to_array(P_bin, idx_test)
        self.report.add_evaluation_data(self.ground_truth.getArrayFromSliceMatrix(
            SparseTensor.CONNECTION_SLICE, idx_test), binary_pred)
        if self.args.statistics:
            write_precision_recall_curve_file(
                self.output_folder + "/statistics/rescal_" + self.start_time,
                "precision_recall_curve_fold%d.csv" % self.foldNumber, precision, recall, threshold)
            TP, FP, threshold = m.roc_curve(self.ground_truth.getArrayFromSliceMatrix(
                SparseTensor.CONNECTION_SLICE, idx_test), prediction)
            write_ROC_curve_file(self.output_folder + "/statistics/rescal_" + self.start_time,
                                 "ROC_curve_fold%d.csv" % self.foldNumber, TP, FP, threshold)
            self.evalDetails.add_statistic_details(self.ground_truth.getSliceMatrix(
                SparseTensor.CONNECTION_SLICE), P_bin, idx_test, prediction)
        self.foldNumber += 1

    def finish_evaluation(self):
        self.AUC_test = np.array(self.AUC_test)
        self.logger.info('AUC-PR Test Mean / Std: %f / %f' % (self.AUC_test.mean(), self.AUC_test.std()))
        self.logger.info('----------------------------------------------------')
        self.log1()
        self.report.summary()
        if self.args.statistics:
            self.evalDetails.output_statistic_details(
                self.output_folder + "/statistics/rescal_" + self.start_time,
                self.ground_truth.getHeaders(), self.args.fbeta, True)
            gexf = create_gexf_graph(self.ground_truth, self.evalDetails)
            output_file = open(self.output_folder + "/statistics/rescal_" + self.start_time + "/graph.gexf", "w")
            gexf.write(output_file)
            output_file.close()

# ========================================================================================
# Implementation of evaluation of RESCAL similarity algorithm
# ========================================================================================
# Notes:
# - changing the rank parameter influences the amount of internal latent "clusters" of the
# algorithm and thus the quality of the matching as well as performance (memory and
# execution time).
# - higher threshold for RESCAL algorithm need similarity means higher recall
# ========================================================================================
class RescalSimilarityEvaluation(EvaluationAlgorithm):

    def __init__(self, args, output_folder, logger, ground_truth, start_time):
        self.init(args, output_folder, logger, ground_truth, start_time)
        self.rank = int(args.rescalsim[0])
        self.threshold = float(args.rescalsim[1])
        self.evalDetails = NeedEvaluationDetailDict()
        self.offers = ground_truth.getOfferIndices()
        self.wants = ground_truth.getWantIndices()
        self.foldNumber = 0

    def log1(self):
        self.logger.info('For RESCAL prediction based on need similarity with threshold: %f' % self.threshold)

    def evaluate_fold(self, test_tensor, test_needs, idx_test):

        # execute the rescal algorithm
        useNeedTypeSlice = (self.args.rescalsim[2] == 'True')
        useConnectionSlice = (self.args.rescalsim[3] == 'True')
        A, R = execute_rescal(test_tensor, self.rank, useNeedTypeSlice, useConnectionSlice)

        # use the most similar needs per need to predict connections
        self.log1()
        P_bin = predict_rescal_connections_by_need_similarity(A, self.threshold, self.offers, self.wants, test_needs)
        binary_pred = matrix_to_array(P_bin, idx_test)
        self.report.add_evaluation_data(self.ground_truth.getArrayFromSliceMatrix(
            SparseTensor.CONNECTION_SLICE, idx_test), binary_pred)

        if self.args.statistics:
            S = similarity_ranking(A)
            y_prop = [1.0 - i for i in np.nan_to_num(S[idx_test])]
            precision, recall, threshold = m.precision_recall_curve(
                self.ground_truth.getArrayFromSliceMatrix(SparseTensor.CONNECTION_SLICE, idx_test), y_prop)
            write_precision_recall_curve_file(
                self.output_folder + "/statistics/rescalsim_" + self.start_time,
                "precision_recall_curve_fold%d.csv" % self.foldNumber, precision, recall, threshold)
            TP, FP, threshold = m.roc_curve(self.ground_truth.getArrayFromSliceMatrix(
                    SparseTensor.CONNECTION_SLICE, idx_test), y_prop)
            write_ROC_curve_file(self.output_folder + "/statistics/rescalsim_" + self.start_time,
                                 "ROC_curve_fold%d.csv" % self.foldNumber, TP, FP, threshold)
            self.evalDetails.add_statistic_details(self.ground_truth.getSliceMatrix(
                SparseTensor.CONNECTION_SLICE), P_bin, idx_test)

    def finish_evaluation(self):
        self.log1()
        self.report.summary()
        if self.args.statistics:
            self.evalDetails.output_statistic_details(
                self.output_folder + "/statistics/rescalsim_" + self.start_time,
                self.ground_truth.getHeaders(), self.args.fbeta)
            gexf = create_gexf_graph(self.ground_truth, self.evalDetails)
            output_file = open(self.output_folder + "/statistics/rescalsim_" + self.start_time + "/graph.gexf", "w")
            gexf.write(output_file)
            output_file.close()


# ========================================================================================
# Implementation of evaluation of cosine similarity algorithm
# ========================================================================================
# Notes:
# higher threshold for cosine similarity link prediction means higher recall.
# set transitive threshold < threshold to avoid transitive predictions.
# ========================================================================================
class CosineEvaluation(EvaluationAlgorithm):

    def __init__(self, args, output_folder, logger, ground_truth, start_time, weighted):
        self.init(args, output_folder, logger, ground_truth, start_time)
        self.weighted = weighted
        self.threshold = float(args.cosine_weigthed[0]) if weighted else float(args.cosine[0])
        self.transitive_threshold = float(args.cosine_weigthed[1]) if weighted else float(args.cosine[1])
        self.evalDetails = NeedEvaluationDetailDict()

    def logEvaluationLine(self):
        str = ""
        if self.weighted:
            str = " weighted"
        self.logger.info('For prediction of%s cosine similarity between needs with thresholds %f, %f:' %
                         (str, self.threshold, self.transitive_threshold))

    def evaluate_fold(self, test_tensor, test_needs, idx_test):
        self.logEvaluationLine()
        binary_pred = cosinus_link_prediciton(test_tensor, test_needs, self.threshold,
                                              self.transitive_threshold, self.weighted)
        self.report.add_evaluation_data(self.ground_truth.getArrayFromSliceMatrix(
            SparseTensor.CONNECTION_SLICE, idx_test), matrix_to_array(binary_pred, idx_test))
        if self.args.statistics:
            self.evalDetails.add_statistic_details(
                self.ground_truth.getSliceMatrix(SparseTensor.CONNECTION_SLICE),
                binary_pred, idx_test)

    def finish_evaluation(self):
        self.logEvaluationLine()
        self.report.summary()
        if self.args.statistics:
            folder = "/statistics/cosine_"
            if self.weighted:
                folder = "/statistics/wcosine_"
            self.evalDetails.output_statistic_details(
                self.output_folder + folder + self.start_time,
                self.ground_truth.getHeaders(), self.args.fbeta)
            gexf = create_gexf_graph(self.ground_truth, self.evalDetails)
            output_file = open(self.output_folder + folder +
                               self.start_time + "/graph.gexf", "w")
            gexf.write(output_file)
            output_file.close()

# ========================================================================================
# Implementation of evaluation of loading an external matrix file with predictions
# ========================================================================================
# Notes:
# Matrix connection file has the same file format as connection slice of tensor
# ========================================================================================
class PredictionMatrixFileEvaluation(EvaluationAlgorithm):

    def __init__(self, args, output_folder, logger, ground_truth, start_time):
        self.init(args, output_folder, logger, ground_truth, start_time)
        header_input = args.inputfolder + "/" + args.headers
        self.file_prediction_tensor = read_input_tensor(
            header_input, [args.prediction_matrix_file], [SparseTensor.CONNECTION_SLICE], True)

    def logEvaluationLine(self):
        self.logger.info('External file (' + self.args.prediction_matrix_file + ') predictions: ')

    def evaluate_fold(self, test_tensor, test_needs, idx_test):
        file_pred = self.file_prediction_tensor.getArrayFromSliceMatrix(
            SparseTensor.CONNECTION_SLICE, idx_test);
        self.report.add_evaluation_data(self.ground_truth.getArrayFromSliceMatrix(
            SparseTensor.CONNECTION_SLICE, idx_test), file_pred)

    def finish_evaluation(self):
        self.logEvaluationLine()
        self.report.summary()

# ======================================================================================
# Implementation of evaluation of combination of algorithms RESCAL and Cosine similarity.
# ======================================================================================
# Notes:
# predict connections by combining the execution of algorithms. First execute the cosine
# similarity algorithm (preferably choosing a threshold to get a high precision) and with
# this predicted matches execute the RESCAL algorithm afterwards (to increase the recall)
# ======================================================================================
class CombineCosineRescalEvaluation(EvaluationAlgorithm):

    def __init__(self, args, output_folder, logger, ground_truth, start_time):
        self.init(args, output_folder, logger, ground_truth, start_time)
        self.report2 = EvaluationReport(logger, args.fbeta)

    def log1(self):
        self.logger.info('First step for prediction of cosine similarity with threshold: %f:' %
                         float(self.args.cosine_rescal[2]))

    def log2(self):
        self.logger.info('And second step for combined RESCAL prediction with parameters: %d, %f:'
                         % (int(self.args.cosine_rescal[0]), float(self.args.cosine_rescal[1])))

    def evaluate_fold(self, test_tensor, test_needs, idx_test):
        cosine_pred, rescal_pred = self.predict_combine_cosine_rescal(
            test_tensor, test_needs, idx_test, int(self.args.cosine_rescal[0]),
            float(self.args.cosine_rescal[1]), float(self.args.cosine_rescal[2]),
            bool(self.args.cosine_rescal[3]))
        self.log1()
        self.report.add_evaluation_data(self.ground_truth.getArrayFromSliceMatrix(
                SparseTensor.CONNECTION_SLICE, idx_test), cosine_pred)
        self.log2()
        self.report2.add_evaluation_data(self.ground_truth.getArrayFromSliceMatrix(
            SparseTensor.CONNECTION_SLICE,idx_test), rescal_pred)

    def finish_evaluation(self):
        self.log1()
        self.report.summary()
        self.log2()
        self.report2.summary()

    # predict connections by combining the execution of algorithms
    def predict_combine_cosine_rescal(self, input_tensor, test_needs, idx_test, rank,
                                      rescal_threshold, cosine_threshold, useNeedTypeSlice):

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


# ======================================================================================
# Implementation of evaluation of intersection of algorithms RESCAL and Cosine similarity.
# ======================================================================================
# Notes:
# predict connections by combining the execution of algorithms. Compute the predictions
# of connections for both cosine similarity and rescal algorithm. Then return the
# intersection of the predictions.
# ======================================================================================
class IntersectionCosineRescalEvaluation(EvaluationAlgorithm):

    def __init__(self, args, output_folder, logger, ground_truth, start_time):
        self.init(args, output_folder, logger, ground_truth, start_time)
        self.report2 = EvaluationReport(logger, args.fbeta)
        self.report3 = EvaluationReport(logger, args.fbeta)

    def log1(self):
        self.logger.info('Intersection of predictions of cosine similarity and rescal algorithms: ')

    def log2(self):
        self.logger.info('For RESCAL prediction with threshold %f:' % float(self.args.intersection[1]))

    def log3(self):
        self.logger.info('For prediction of cosine similarity between needs with thresholds: %f:' %
                  float(self.args.intersection[2]))

    def evaluate_fold(self, test_tensor, test_needs, idx_test):
        inter_pred, cosine_pred, rescal_pred = self.predict_intersect_cosine_rescal(
            test_tensor, test_needs, idx_test, int(self.args.intersection[0]),
            float(self.args.intersection[1]), float(self.args.intersection[2]),
            bool(self.args.intersection[3]))
        self.log1()
        self.report.add_evaluation_data(self.ground_truth.getArrayFromSliceMatrix(
            SparseTensor.CONNECTION_SLICE, idx_test), inter_pred)
        self.log2()
        self.report2.add_evaluation_data(self.ground_truth.getArrayFromSliceMatrix(
            SparseTensor.CONNECTION_SLICE,idx_test), rescal_pred)
        self.log3()
        self.report3.add_evaluation_data(self.ground_truth.getArrayFromSliceMatrix(
            SparseTensor.CONNECTION_SLICE,idx_test), cosine_pred)

    def finish_evaluation(self):
        self.log1()
        self.report.summary()
        self.log2()
        self.report2.summary()
        self.log3()
        self.report3.summary()

    # predict connections by intersection of RESCAL and cosine results
    def predict_intersect_cosine_rescal(self, input_tensor, test_needs, idx_test, rank,
                                        rescal_threshold, cosine_threshold, useNeedTypeSlice):

        wants = input_tensor.getWantIndices()
        offers = input_tensor.getOfferIndices()

        # execute the cosine algorithm
        binary_pred_cosine = cosinus_link_prediciton(input_tensor, test_needs, cosine_threshold, 0.0, False)

        # execute the rescal algorithm
        A,R = execute_rescal(input_tensor, rank)
        P_bin = predict_rescal_connections_by_threshold(A, R, rescal_threshold, offers, wants, test_needs)

        # return the intersection of the prediction of both algorithms
        binary_pred_cosine = matrix_to_array(binary_pred_cosine, idx_test)
        binary_pred_rescal = matrix_to_array(P_bin, idx_test)
        binary_pred = [min(binary_pred_cosine[i], binary_pred_rescal[i]) for i in range(len(binary_pred_cosine))]
        return binary_pred, binary_pred_cosine, binary_pred_rescal
