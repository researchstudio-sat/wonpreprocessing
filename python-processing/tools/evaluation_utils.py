import codecs
import os

__author__ = 'hfriedrich'

import numpy as np
from tensor_utils import matrix_to_array
import sklearn.metrics as m

# class to store statistical detail data for a need, data like number true positives, true negatives,
# false positives, false negatives can be used to calculate precision, recall, accuracy, fscore.
# Also for a need it can store all TP, FN, FP indices to all other needs of the evaluation. This can be used for very
# detailed analysis.
class NeedEvaluationDetails:

    def __init__(self, need):
        self.need = need
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.TP_toNeeds = []
        self.FN_toNeeds = []
        self.FP_toNeeds = []
        self.TP_thresholds = []
        self.FN_thresholds = []
        self.FP_thresholds = []

    def addClassificationData(self, y_true, y_pred, toNeeds=[], thresholds=[]):
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                if y_true[i] == 1.0:
                    self.TP += 1
                    if len(toNeeds) > 0:
                        self.TP_toNeeds.append(toNeeds[i])
                        if len(thresholds) > 0:
                            self.TP_thresholds.append(thresholds[i])
                else:
                    self.TN += 1
            else:
                if y_true[i] == 1.0:
                    self.FN += 1
                    if len(toNeeds) > 0:
                        self.FN_toNeeds.append(toNeeds[i])
                        if len(thresholds) > 0:
                            self.FN_thresholds.append(thresholds[i])
                else:
                    self.FP += 1
                    if len(toNeeds) > 0:
                        self.FP_toNeeds.append(toNeeds[i])
                        if len(thresholds) > 0:
                            self.FP_thresholds.append(thresholds[i])

    def getPrecision(self):
        sum = self.TP + self.FP
        return self.TP / float(sum) if sum > 0 else 1.0

    def getRecall(self):
        sum = self.TP + self.FN
        return self.TP / float(sum) if sum > 0 else 1.0

    def getAccuracy(self):
        sum = self.TP + self.TN + self.FP + self.FN
        return (self.TP + self.TN) / float(sum) if sum > 0 else 1.0

    def getFScore(self, f_beta):
        div = (f_beta * f_beta * self.getPrecision() + self.getRecall())
        f_score = 0.0
        if div != 0:
            f_score = (1 + f_beta * f_beta) * (self.getPrecision() * self.getRecall()) / div
        return f_score

# this class is basically a dictionary of statistical need detail data for each need. Use this to store all data per
# need for the whole evaluation
class NeedEvaluationDetailDict:

    def __init__(self):
        self.dict = dict()

    def retrieveNeedDetails(self, need):
        if (need not in self.dict):
            self.dict[need] = NeedEvaluationDetails(need)
        return self.dict[need]

    def add_statistic_details(self, con_slice_true, con_slice_pred, idx_test, thresholds=[]):
        sorted_idx = np.argsort(idx_test[0])
        i1 = [idx_test[0][i] for i in sorted_idx]
        i2 = [idx_test[1][i] for i in sorted_idx]
        if len(thresholds) > 0:
            sorted_thresholds = [thresholds[i] for i in sorted_idx]
        idx_test = (i1, i2)

        from_idx = 0
        while from_idx < len(idx_test[0]):
            need_from = idx_test[0][from_idx]
            to_idx = np.searchsorted(idx_test[0][from_idx:], need_from, side='right') + from_idx
            needDetails = self.retrieveNeedDetails(need_from)
            idx_temp = (idx_test[0][from_idx:to_idx], idx_test[1][from_idx:to_idx])
            th = None
            if len(thresholds) > 0:
                th = sorted_thresholds[from_idx:to_idx]
            needDetails.addClassificationData(matrix_to_array(con_slice_true, idx_temp),
                                              matrix_to_array(con_slice_pred, idx_temp), idx_temp[1], th)
            from_idx = to_idx

    # helper function
    def create_file_from_sorted_list(self, dir, filename, list):
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
    def output_statistic_details(self, outputpath, headers, fbeta, printThresholds=False):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        summary_file = codecs.open(outputpath + "/_summary.txt",'a+',encoding='utf8')

        for need in self.dict.keys():
            # write need details file
            needEval = self.dict[need]
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

            self.create_file_from_sorted_list(outputpath, headers[need][6:] + ".txt", needDetails)

        # write the summary file
        summary_file.write(headers[need][6:])
        summary_file.write(": TP: " + str(self.dict[need].TP))
        summary_file.write(": TN: " + str(self.dict[need].TN))
        summary_file.write(": FP: " + str(self.dict[need].FP))
        summary_file.write(": FN: " + str(self.dict[need].FN))
        summary_file.write(": Precision: " + str(self.dict[need].getPrecision()))
        summary_file.write(": Recall: " + str(self.dict[need].getRecall()))
        summary_file.write(": f%f-score : " % fbeta + str(self.dict[need].getFScore(fbeta)))
        summary_file.write(": Accuracy: " + str(self.dict[need].getAccuracy()) + "\n")
        summary_file.close()




# class to collect data during the runs of the test and print calculated measures for summary
class EvaluationReport:

    def __init__(self, logger, f_beta=1.0):
        self.f_beta = f_beta
        self.precision = []
        self.recall = []
        self.accuracy = []
        self.fscore = []
        self.logger = logger

    def add_evaluation_data(self, y_true, y_pred):
        p, r, f, _ =  m.precision_recall_fscore_support(y_true, y_pred, average='weighted', beta=self.f_beta)
        a = m.accuracy_score(y_true, y_pred)
        cm = m.confusion_matrix(y_true, y_pred, [1, 0])
        self.precision.append(p)
        self.recall.append(r)
        self.fscore.append(f)
        self.accuracy.append(a)
        self.logger.info('accuracy: %f' % a)
        self.logger.info('precision: %f' % p)
        self.logger.info('recall: %f' % r)
        self.logger.info('f%.01f-score: %f' % (self.f_beta, f))
        self.logger.info('confusion matrix: ' + str(cm))

    def summary(self):
        a = np.array(self.accuracy)
        p = np.array(self.precision)
        r = np.array(self.recall)
        f = np.array(self.fscore)
        self.logger.info('Accuracy Mean / Std: %f / %f' % (a.mean(), a.std()))
        self.logger.info('Precision Mean / Std: %f / %f' % (p.mean(), p.std()))
        self.logger.info('Recall Mean / Std: %f / %f' % (r.mean(), r.std()))
        self.logger.info('F%.01f-Score Mean / Std: %f / %f' % (self.f_beta, f.mean(), f.std()))


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

# write precision/recall (and threshold) curve to file
def write_precision_recall_curve_file(folder, outfilename, precision, recall, threshold):
    if not os.path.exists(folder):
        os.makedirs(folder)
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
    file = codecs.open(folder + "/" + outfilename,'w+',encoding='utf8')
    file.write("TP, FP, threshold")
    prevline = ""
    for i in range(1, len(threshold)):
        line = "\n%.3f, %.3f, %.3f" % (TP[i], FP[i], threshold[i])
        if line != prevline:
            file.write(line)
            prevline = line
    file.close()
