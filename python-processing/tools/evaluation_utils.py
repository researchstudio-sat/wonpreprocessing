__author__ = 'hfriedrich'

import numpy as np
from tensor_utils import matrix_to_array

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

    def addClassificationData(self, y_true, y_pred, toNeeds=None, thresholds=None):
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                if y_true[i] == 1.0:
                    self.TP += 1
                    if toNeeds != None:
                        self.TP_toNeeds.append(toNeeds[i])
                        if thresholds != None:
                            self.TP_thresholds.append(thresholds[i])
                else:
                    self.TN += 1
            else:
                if y_true[i] == 1.0:
                    self.FN += 1
                    if toNeeds != None:
                        self.FN_toNeeds.append(toNeeds[i])
                        if thresholds != None:
                            self.FN_thresholds.append(thresholds[i])
                else:
                    self.FP += 1
                    if toNeeds != None:
                        self.FP_toNeeds.append(toNeeds[i])
                        if thresholds != None:
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

    def add_statistic_details(self, con_slice_true, con_slice_pred, idx_test, thresholds=None):
        sorted_idx = np.argsort(idx_test[0])
        i1 = [idx_test[0][i] for i in sorted_idx]
        i2 = [idx_test[1][i] for i in sorted_idx]
        if thresholds != None:
            sorted_thresholds = [thresholds[i] for i in sorted_idx]
        idx_test = (i1, i2)

        from_idx = 0
        while from_idx < len(idx_test[0]):
            need_from = idx_test[0][from_idx]
            to_idx = np.searchsorted(idx_test[0][from_idx:], need_from, side='right') + from_idx
            needDetails = self.retrieveNeedDetails(need_from)
            idx_temp = (idx_test[0][from_idx:to_idx], idx_test[1][from_idx:to_idx])
            th = None
            if thresholds != None:
                th = sorted_thresholds[from_idx:to_idx]
            needDetails.addClassificationData(matrix_to_array(con_slice_true, idx_temp),
                                              matrix_to_array(con_slice_pred, idx_temp), idx_temp[1], th)
            from_idx = to_idx

