__author__ = 'hfriedrich'

import os
from gexf import Gexf
from time import strftime
from tensor_utils import SparseTensor
from evaluation_utils import NeedEvaluationDetailDict, NeedEvaluationDetails

# create a gexf graph from the tensor for visualization in gephi
# add the following data:
# - needs (nodes)
# - connections (edges)
# - need labels & ids
# - need type
# - need attributes (subject & content keywords, categories)
def create_gexf_graph(tensor, needEvaluationDetailDict=None):
    needs = tensor.getNeedIndices()
    offers = tensor.getOfferIndices()
    wants = tensor.getWantIndices()
    date_time = strftime("%Y-%m-%d_%H%M%S")

    gexf = Gexf(os.path.basename(__file__), date_time)
    graph = gexf.addGraph('undirected','static','generated need graph')
    need_type_attr = graph.addNodeAttribute("need type", "undefined", "string")
    subject_attr = graph.addNodeAttribute("subject attributes", "", "string")
    content_attr = graph.addNodeAttribute("content attributes", "", "string")
    category_attr = graph.addNodeAttribute("category attributes", "", "string")

    if needEvaluationDetailDict:
        TP_attr = graph.addNodeAttribute("TP", "", "integer")
        TN_attr = graph.addNodeAttribute("TN", "", "integer")
        FP_attr = graph.addNodeAttribute("FP", "", "integer")
        FN_attr = graph.addNodeAttribute("FN", "", "integer")
        precision_attr = graph.addNodeAttribute("precision", "", "float")
        recall_attr = graph.addNodeAttribute("recall", "", "float")
        accuracy_attr = graph.addNodeAttribute("accuracy", "", "float")
        f0_5score_attr = graph.addNodeAttribute("f0.5score", "", "float")
        f1score_attr = graph.addNodeAttribute("f1score", "", "float")

    # add the nodes
    for need in needs:
        # add a node for each need
        node = graph.addNode(need, tensor.getNeedLabel(need))

        # set the need type to each node as an attribute
        if need in offers:
            node.addAttribute(need_type_attr, "OFFER")
        if need in wants:
            node.addAttribute(need_type_attr, "WANT")

        # get the attributes for each need
        attr = tensor.getAttributesForNeed(need, SparseTensor.ATTR_SUBJECT_SLICE)
        node.addAttribute(subject_attr, ', '.join(attr))
        attr = tensor.getAttributesForNeed(need, SparseTensor.ATTR_CONTENT_SLICE)
        node.addAttribute(content_attr, ', '.join(attr))
        attr = tensor.getAttributesForNeed(need, SparseTensor.CATEGORY_SLICE)
        attr.sort()
        node.addAttribute(category_attr, ', '.join(attr))

        # add the statistical detail data to every node
        if needEvaluationDetailDict:
            needDetail = needEvaluationDetailDict.retrieveNeedDetails(need)
            node.addAttribute(TP_attr, str(needDetail.TP))
            node.addAttribute(TN_attr, str(needDetail.TN))
            node.addAttribute(FN_attr, str(needDetail.FN))
            node.addAttribute(FP_attr, str(needDetail.FP))
            node.addAttribute(precision_attr, str(needDetail.getPrecision()))
            node.addAttribute(recall_attr, str(needDetail.getRecall()))
            node.addAttribute(accuracy_attr, str(needDetail.getAccuracy()))
            node.addAttribute(f0_5score_attr, str(needDetail.getFScore(0.5)))
            node.addAttribute(f1score_attr, str(needDetail.getFScore(1)))

    # add the connections as edges between nodes (needs)
    nz = tensor.getSliceMatrix(SparseTensor.CONNECTION_SLICE).nonzero()
    for i in range(len(nz[0])):
        graph.addEdge(str(nz[0][i]) + "_" + str(nz[1][i]), nz[0][i], nz[1][i])

    return gexf

