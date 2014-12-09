library("igraph")
library("Matrix")
library("rgexf")
library("XML")


# This script loads some the output of the rescal tensor processing by the python program 
# "generate_rescal_output.py". It loads the data from the connection slice of the tensor
# before and after execution of the rescal algorithm to compute graph measures based on
# this data including graphics.

# function to delete isolate nodes from the graph
deleteIsolates <- function(x) {
  isolates <- which(degree(x) == 0)
  return (delete.vertices(x, isolates))
}

# function to concat two strings
concat <- function(s1, s2) {
  return (paste(c(s1,s2), collapse=''))
}

# Converts the given igraph object to GEXF format and saves it at the given filepath location
#     g: input igraph object to be converted to gexf format
#     filepath: file location where the output gexf file should be saved
#
saveAsGEXF = function(g, filepath="converted_graph.gexf")
{
  require(igraph)
  require(rgexf)
  
  # gexf nodes require two column data frame (id, label)
  # check if the input vertices has label already present
  # if not, just have the ids themselves as the label
  if(is.null(V(g)$label))
    V(g)$label <- as.character(V(g))
  
  # similarily if edges does not have weight, add default 1 weight
  if(is.null(E(g)$weight))
    E(g)$weight <- rep.int(1, ecount(g))
  
  nodes <- data.frame(cbind(V(g), V(g)$label))
  edges <- t(Vectorize(get.edge, vectorize.args='id')(g, 1:ecount(g)))
  
  # combine all node attributes into a matrix (and take care of & for xml)
  vAttrNames <- setdiff(list.vertex.attributes(g), "label") 
  nodesAtt <- data.frame(sapply(vAttrNames, function(attr) sub("&", "&",get.vertex.attribute(g, attr))))
  
  # combine all edge attributes into a matrix (and take care of & for xml)
  eAttrNames <- setdiff(list.edge.attributes(g), "weight") 
  edgesAtt <- data.frame(sapply(eAttrNames, function(attr) sub("&", "&",get.edge.attribute(g, attr))))
  
  # combine all graph attributes into a meta-data
  graphAtt <- sapply(list.graph.attributes(g), function(attr) sub("&", "&",get.graph.attribute(g, attr)))
  
  # generate the gexf object
  output <- write.gexf(nodes, edges, 
                       edgesWeight=E(g)$weight,
                       edgesAtt = edgesAtt,
                       nodesAtt = nodesAtt,
                       meta=c(list(creator="Gopalakrishna Palem", description="igraph -> gexf converted file", keywords="igraph, gexf, R, rgexf"), graphAtt))
  
  print(output, filepath, replace=T)
}

# load the input data files
# headers: headers of the tensor, need and attribute names
# incon: input connection slice of the tensor to the rescal algorithm
# outcon: output (predicted) connection slice of the rescal algorithm
# needtype: need type slice of the tensor
dataFolder <- "C:/dev/temp/testdataset_20141112/evaluation/tensor/"
headers <- readLines(concat(dataFolder,"headers.txt"))
incon <- readMM(concat(dataFolder,"connection.mtx"))
outcon <- readMM(concat(dataFolder,"connection.mtx"))
needtype <- readMM(concat(dataFolder,"needtype.mtx"))

# create the graphs
igraph.options(vertex.size=3, vertex.label=NA)
g_incon <- graph.adjacency(incon, "undirected")
g_outcon <- graph.union(graph.adjacency(outcon, "undirected"), g_incon)
g_type <- graph.adjacency(needtype, "directed")
indices <- which(headers != "")
offerNodes <- get.edges(g_type, E(g_type)[to(which(headers=="Attr: OFFER"))])
wantNodes <- get.edges(g_type, E(g_type)[to(which(headers=="Attr: WANT"))])
types <- ifelse(is.element(indices, offerNodes), "OFFER", ifelse(is.element(indices, wantNodes), "WANT", "UNDEFINED"))
nodes <- data.frame(indices, need=headers, type=types)
edges_in <- get.data.frame(g_incon, what=("edges"))
edges_out <- get.data.frame(g_outcon, what=("edges"))
g_incon <- graph.data.frame(edges_in, vertices=nodes, directed=FALSE)
g_outcon <- graph.data.frame(edges_out, vertices=nodes, directed=FALSE)
attrVertices <- which(substr(V(g_incon)$need,0,5) == "Attr:")
g_incon <- delete.vertices(g_incon, attrVertices)
g_outcon <- delete.vertices(g_outcon, attrVertices)
g_incon_all <- g_incon
g_outcon_all <- g_outcon
g_incon <- deleteIsolates(g_incon)
g_outcon <- deleteIsolates(g_outcon)
g_union <- graph.union(g_incon, g_outcon)
V(g_union)$need <- ifelse(is.na(V(g_union)$need_1), V(g_union)$need_2, V(g_union)$need_1)
V(g_union)$type <- ifelse(is.na(V(g_union)$type_1), V(g_union)$type_2, V(g_union)$type_1)
V(g_incon)$color <- ifelse(V(g_incon)$type=="OFFER", "blue", ifelse(V(g_incon)$type=="WANT", "red", "green"))
V(g_union)$color <- ifelse(V(g_union)$type=="OFFER", "blue", ifelse(V(g_union)$type=="WANT", "red", "green"))
V(g_union)$value <- 1.0

# print a summary of the graph measures
cat("Input connections graph summary:", "\nNumber of Needs: ", vcount(g_incon_all),
    "( OFFERS:", length(which(V(g_incon_all)$type == "OFFER")),
    "WANTS:", length(which(V(g_incon_all)$type == "WANT")), 
    "UNDEFINED:", length(which(V(g_incon_all)$type == "UNDEFINED")), ")",
    "\nNumber of Needs with connections:", vcount(g_incon),
    "( OFFERS:", length(which(V(g_incon)$type == "OFFER")),
    "WANTS:", length(which(V(g_incon)$type == "WANT")), 
    "UNDEFINED:", length(which(V(g_incon)$type == "UNDEFINED")), ")",
    "\nNumber of Connections:",ecount(g_incon))

cat("Predicted (+input) connections graph summary:", "\nNumber of Needs: ", vcount(g_incon_all),
    "( OFFERS:", length(which(V(g_incon_all)$type == "OFFER")),
    "WANTS:", length(which(V(g_incon_all)$type == "WANT")), 
    "UNDEFINED:", length(which(V(g_incon_all)$type == "UNDEFINED")), ")",
    "\nNumber of Needs with connections:", vcount(g_union),
    "( OFFERS:", length(which(V(g_union)$type == "OFFER")),
    "WANTS:", length(which(V(g_union)$type == "WANT")), 
    "UNDEFINED:", length(which(V(g_union)$type == "UNDEFINED")), ")",
    "\nNumber of Connections:",ecount(g_union))

# plot histograms for degree
histin <- hist(degree(g_incon), breaks=max(degree(g_incon)), main="Input Connection Histogram", xlab="number of needs", ylab="connections per need")
Sys.sleep(5)
histout <- hist(degree(g_union), breaks=max(degree(g_union)), main="Predicted (+Input) Connection Histogram", xlab="number of needs", ylab="connections per need")

# create the layout for the graphs
layall <- layout.auto(g_union)
layall <- layout.norm(layall, xmin = -1, xmax = 1, ymin = -1, ymax = 1)
layin <- layall[which(is.element(V(g_incon)$name, V(g_union)$name)),]

# plot the graphs after another to see the difference 
# (execute the commands manually to export pictures of the graphs)
plot(g_incon, layout=layin, edge.curved=TRUE)
Sys.sleep(5)
plot(g_union, layout=layall, edge.curved=TRUE)

# save the graph for use in gephi
saveAsGEXF(g_union, "testgraph3.gexf")


# use this to select specific nodes and see their need name
target <- identify(layall[,1], layall[,2])
V(g_union)[target]$need
