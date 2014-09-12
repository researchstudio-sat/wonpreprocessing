
__author__ = 'bivanschitz'

from scipy.spatial.distance import pdist, cosine, matching, dice
from math import log10

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
def add_connections(candidates, connectionmatrix, new_element_index,checkset):
    for item in candidates:
        postion = int(item[0])
        if connectionmatrix[new_element_index,postion] == 0 and postion in checkset:
            connectionmatrix[new_element_index,postion] = 1
            continue
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

# the cosine link prediction algorithm
def cosinus_link_prediciton(tensormatrix, offers, wants, new_elements, simlimit):

    # slice 2 of the tensor are the attributes
    attributemat = tensormatrix[2]
    attributemat=attributemat.toarray()

    # slice 0 of the tensor are the connections
    connectionmat = tensormatrix[0]
    connectionmat = connectionmat.toarray()

    for new_element in new_elements:
        if new_element in offers:
            checkset = wants
        else:
            checkset = offers

        #get the most commen elements
        most_commen_elements = most_common_elements(checkset,attributemat,new_element)

        #get the candidates for the link prediction
        candidates = get_candidates(most_commen_elements,simlimit)
        newconnectionmat = add_connections(candidates, connectionmat,new_element,checkset)

    return newconnectionmat

# the weighted cosine link prediction algorithm
def cosinus_weighted_link_prediction(tensormatrix, offers, wants, new_elements, simlimit):

    # slice 2 of the tensor are the attributes
    attributemat = tensormatrix[2]
    attributemat=attributemat.toarray()

    # slice 0 of the tensor are the connections
    connectionmat = tensormatrix[0]
    connectionmat = connectionmat.toarray()

    for new_element in new_elements:
        if new_element in offers:
            checkset = wants
        else:
            checkset = offers

        #get the weighted attribut matrix
        weighted= termFrequencies(attributemat)
        weightedmat = weighted * attributemat


        #get the most comment elements
        most_common_elements_weighted = most_common_elements(checkset,weightedmat , new_element)
        #get the candidates for the link prediction
        candidates = get_candidates(most_common_elements_weighted,simlimit)
        newconnectionmat = add_connections(candidates, connectionmat,new_element,checkset)

    return newconnectionmat

