#!/usr/bin/env python

import logging

from src.main.python import evaluate_rescal as util

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Mail Example')

import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import codecs

# write a file with the top X term attribute predictions for every need
def write_term_output(file, predicted_tensor, headers):
    TOPX = 10
    needs = util.need_indices(headers)
    _log.info('Writing term attribute prediction output file: ' + file)
    out = codecs.open(file, 'w+', encoding='utf8')
    for need in needs:
        darr = np.array(predicted_tensor[need,:,2])
        indices = (np.argsort(darr))[-TOPX:]
        predicted = [headers[i][6:] + " (" + str(round(darr[i], 2)) + ")" for i in reversed(indices)]
        out.write(headers[need].ljust(150) + ': ' + ', '.join(predicted) + '\n')
    out.close()

# write a file with the original and predicted connections between each need
def write_connection_output(file, input_tensor, predicted_tensor, headers):
    _log.info('Writing connection prediction output file: ' + file)
    out = codecs.open(file, 'w+', encoding='utf8')
    needs = util.need_indices(headers)
    for from_need in needs:
        darr = np.array(predicted_tensor[from_need,:,0])
        indices = reversed((np.argsort(darr))[-20:])
        indices = [i for i in indices if darr[i] > 0.1 and i in needs]
        if len(indices) > 0:
            out.write('\n')
            out.write(headers[from_need][6:] + '\n')
        for to_need in indices:
            newPrediction = ("NEW_PREDICTION: " if input_tensor[0].getrow(from_need).getcol(to_need)[0,0] == 0.0
                             else "")
            predicted_entities = newPrediction + headers[to_need] + " (" + str(round(darr[i], 2)) + ")"
            out.write(predicted_entities + '\n')
    out.close()

# write a file with the top X most similar needs for each need
def write_need_output(file, similarity_matrix, headers):
    TOPX = 20
    _log.info('Writing need similarity output file: ' + file)
    out = codecs.open(file, 'w+', encoding='utf8')
    needs = util.need_indices(headers)
    for from_need in needs:
        indices = (np.argsort(similarity_matrix[from_need,:]))[:TOPX]
        predicted_entities = [headers[i][6:] + " (" + str(round(similarity_matrix[from_need,i], 4)) + ")" for i in
                              indices if i in needs and i != from_need]
        out.write(headers[from_need][6:] + '\n' + '\n'.join(predicted_entities) + '\n\n')


# This program executes the rescal algorithm and creates several output files to
# demonstrate some use cases of the algorithm in conjunction with the mail processing.
# Furthermore mtx output files are generated which represent slices of either the input
# or output tensor. These files can be used to do further evaluation in R.
#
# Input parameters:
# argv[1]: folder with the following files:
# - tensor matrix data file name
# - headers file name
# - connections file name
if __name__ == '__main__':

    # load the tensor input data
    folder = sys.argv[1]
    input_tensor, headers = util.read_input_tensor(folder)
    needs = util.need_indices(headers)
    offers = util.offer_indices(input_tensor, headers)
    wants = util.want_indices(input_tensor, headers)

    # execute rescal algorithm
    RANK = 100
    P, A, R = util.predict_rescal_als(input_tensor, RANK)

    # write output files
    write_term_output(folder + "/outterm.txt", P, headers)
    write_connection_output(folder + "/outconn.txt", input_tensor, P, headers)

    # write output file for further R processing
    P_bin = util.predict_connections_by_threshold(P, 0.05, offers, wants, needs)
    _log.info('Writing predicted connection slice output file: ' + folder + "/outcon.mtx")
    mmwrite(folder + "/outcon.mtx", csr_matrix(P_bin[:,:,0]))

    # write need similarity output file - use only attribute slice, not connection or classification
    P, A, R = util.predict_rescal_als([input_tensor[2]], RANK)
    S = util.similarity_ranking(A)
    write_need_output(folder + "/outneed.txt", S, headers)








