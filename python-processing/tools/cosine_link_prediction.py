
__author__ = 'bivanschitz'

from math import log10

from scipy.spatial.distance import cosine

from tools.tensor_utils import SparseTensor


#FUNCTIONS

#get the most commen elements using the cosinus distance (alternative distance measure available)
def most_common_elements(needindex ,mat, newelement):
    most_common = {}
    for item in needindex:
         if (mat[item].sum() > 0.0 and mat[newelement].sum() > 0.0):
            temp = cosine(mat[item],mat[newelement])
            most_common.update({item: temp})
    result = sorted(most_common.items(), key=lambda x: x[1])
    return result

#Get the candidates witch value is smaller then the bound
def get_candidates(most_common, max_value):
    add_connection = []
    for item in most_common:
        if (float(item[1])) < max_value:
            add_connection.append(item)
            continue
    return add_connection

#add the new connections to the matrix
def add_transitv_connections(candidates, connectionmatrix, new_element_index, checkset, threshold):
    for item in candidates:
        value = item[1]
        position = int(item[0])
        if connectionmatrix[new_element_index, position] == 0 and position in checkset:
            connectionmatrix[new_element_index, position] = 1
        else:
            if (value < threshold) and (position != new_element_index):
                for i, j in enumerate(connectionmatrix[position]):
                    if j == 1 and connectionmatrix[new_element_index, i] == 0:
                        connectionmatrix[new_element_index, i] = 1

    return connectionmatrix

#Gereate the inverse term frequencies
def termFrequencies (attributemat):
    colsum=attributemat.sum(axis=0)
    totalnumberatt=sum(colsum)
    numberdocuments = len(colsum)
    inftermfre = []
    for item in colsum:
        if item == 0:
            inftermfre.append(0)
        else:
            idf = log10(numberdocuments/item)
            inftermfre.append(idf)
    return inftermfre



#############################
#Generate the link prediction

# the cosinus transitiv weighted link prediction algorithm
#
# parameters:
# ============
# tensormatrix: list of csr_matrix slices describing the tensor
# offers: indices in the tensor that references all the offers
# wants: indices in the tensor that references all the wants
# new_elements: list of need indices for which the prediction should be calculated
# threshold: if similarity value between an offer and want is lower than threshold then there is a connection predicted
# transitive_threshold: if similarity between want/want or offer/offer pairs is lower than "threshold"
#   connections to transitive connected needs are taken if their need similarity is lower than "transitive_threshold" in
#   comparison to the origin need. To get transitive predictions set "transitive_threshold" > "threshold" (e.g. set
#   "transitive_threshold" value to 0 for no transitive connection prediction).
# weighted: True if the attribute terms should be weighted
def cosinus_link_prediciton(tensor, new_elements, threshold, transitive_threshold, weighted):
    # slice 2 of the tensor are the attributes
    attributemat = tensor.getSliceMatrix(SparseTensor.ATTR_SUBJECT_SLICE)
    attributemat = attributemat.toarray()
    allneeds = tensor.getNeedIndices()
    offers = tensor.getOfferIndices()
    wants = tensor.getWantIndices()

    # slice 0 of the tensor are the connections
    connectionmat = tensor.getSliceMatrix(SparseTensor.CONNECTION_SLICE)
    connectionmat = connectionmat.toarray()

    for new_element in new_elements:
        if new_element in offers:
            checkset = wants
        else:
            checkset = offers

        # get the weighted attribut matrix
        if weighted:
            weighted = termFrequencies(attributemat)
            attributemat = weighted * attributemat

        #get the most comment elements
        most_common_elements_weighted = most_common_elements(allneeds, attributemat, new_element)
        #get the candidates for the link prediction
        candidates = get_candidates(most_common_elements_weighted, threshold)
        newconnectionmat = add_transitv_connections(candidates, connectionmat, new_element, checkset, transitive_threshold)

    return newconnectionmat