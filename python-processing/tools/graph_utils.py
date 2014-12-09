__author__ = 'hfriedrich'

import os
from gexf import Gexf
from time import strftime
from tensor_utils import SparseTensor

# create a gexf graph from the tensor for visualization in gephi
# add the following data:
# - needs (nodes)
# - connections (edges)
# - need labels & ids
# - need type
# - need attributes (subject & content keywords, categories)
def create_gexf_graph(tensor):
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

    # add the nodes
    for need in needs:
        # add a node for each need
        n = graph.addNode(need, tensor.getNeedLabel(need))

        # set the need type to each node as an attribute
        if need in offers:
            n.addAttribute(need_type_attr, "OFFER")
        if need in wants:
            n.addAttribute(need_type_attr, "WANT")

        # get the attributes for each need
        attr = tensor.getAttributesForNeed(need, SparseTensor.ATTR_SUBJECT_SLICE)
        n.addAttribute(subject_attr, ', '.join(attr))
        attr = tensor.getAttributesForNeed(need, SparseTensor.ATTR_CONTENT_SLICE)
        n.addAttribute(content_attr, ', '.join(attr))
        attr = tensor.getAttributesForNeed(need, SparseTensor.CATEGORY_SLICE)
        attr.sort()
        n.addAttribute(category_attr, ', '.join(attr))

    # add the connections as edges between nodes (needs)
    nz = tensor.getSliceMatrix(SparseTensor.CONNECTION_SLICE).nonzero()
    for i in range(len(nz[0])):
        graph.addEdge(str(nz[0][i]) + "_" + str(nz[1][i]), nz[0][i], nz[1][i])

    return gexf

