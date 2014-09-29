#!/usr/bin/env python

__author__ = 'hfriedrich'

import logging

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Mail Example')

import sys
import codecs
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from tensor_utils import CONNECTION_SLICE, ATTR_SUBJECT_SLICE, need_indices, offer_indices, want_indices, \
    read_input_tensor, predict_rescal_als, predict_rescal_connections_by_threshold, similarity_ranking, predict_rescal_connections_by_need_similarity


# write a file with the top X term attribute predictions for every need
def write_term_output(file, predicted_tensor, headers):
    TOPX = 10
    needs = need_indices(headers)
    _log.info('Writing term attribute prediction output file: ' + file)
    out = codecs.open(file, 'w+', encoding='utf8')
    for need in needs:
        darr = np.array(predicted_tensor[need,:,ATTR_SUBJECT_SLICE])
        indices = (np.argsort(darr))[-TOPX:]
        predicted = [headers[i][6:] + " (" + str(round(darr[i], 2)) + ")" for i in reversed(indices)]
        out.write(headers[need].ljust(150) + ': ' + ', '.join(predicted) + '\n')
    out.close()

# write a file with the original and predicted connections between each need
def write_connection_output(file, input_tensor, predicted_connections, headers):
    _log.info('Writing connection prediction output file: ' + file)
    out = codecs.open(file, 'w+', encoding='utf8')
    needs = need_indices(headers)
    for from_need in needs:
        darr = np.array(predicted_connections[from_need,:])
        indices = [i for i in needs if darr[i] == 1.0 and input_tensor[CONNECTION_SLICE].getrow(from_need).getcol(i)[0,0] == 0.0]
        if len(indices) > 0:
            out.write('\n')
            out.write(headers[from_need][6:] + '\n')
        for to_need in indices:
            out.write(headers[to_need][6:] + '\n')
    out.close()

# write a file with the top X most similar needs for each need
def write_need_output(file, similarity_matrix, headers):
    TOPX = 20
    _log.info('Writing need similarity output file: ' + file)
    out = codecs.open(file, 'w+', encoding='utf8')
    needs = need_indices(headers)
    for from_need in needs:
        indices = (np.argsort(similarity_matrix[from_need,:]))[:TOPX]
        predicted_entities = [headers[i][6:] + " (" + str(round(similarity_matrix[from_need,i], 4)) + ")" for i in
                              indices if i in needs and i != from_need]
        out.write(headers[from_need][6:] + '\n' + '\n'.join(predicted_entities) + '\n\n')
    out.close()

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
    data_input = [folder + "/data-0.mtx", folder + "/data-1.mtx", folder + "/data-2.mtx"]
    header_input = folder + "/headers.txt"
    input_tensor, headers = read_input_tensor(header_input, data_input)
    needs = need_indices(headers)
    offers = offer_indices(input_tensor, headers)
    wants = want_indices(input_tensor, headers)

    # execute rescal algorithm
    RANK = 50
    P, A, R = predict_rescal_als(input_tensor, RANK)

    # write output files
    write_term_output(folder + "/outterm.txt", P, headers)

    # write output file for further R processing
    # P_bin = predict_rescal_connections_by_threshold(P, 0.05, offers, wants, needs)
    P, A, R = predict_rescal_als([input_tensor[CONNECTION_SLICE], input_tensor[ATTR_SUBJECT_SLICE]], RANK)
    P_bin = predict_rescal_connections_by_need_similarity(P, A, 0.005, offers, wants, needs)
    _log.info('Writing predicted connection slice output file: ' + folder + "/outcon.mtx")
    mmwrite(folder + "/outcon.mtx", csr_matrix(P_bin[:,:,CONNECTION_SLICE]))
    write_connection_output(folder + "/outconn.txt", input_tensor, P_bin[:,:,CONNECTION_SLICE], headers)

    # write need similarity output file - use only attribute slice, not connection or classification
    P, A, R = predict_rescal_als([input_tensor[2]], RANK)
    S = similarity_ranking(A)
    write_need_output(folder + "/outneed.txt", S, headers)








