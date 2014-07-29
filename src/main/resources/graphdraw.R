library("igraph")
library("Matrix")

deleteIsolates <- function(x) {
  isolates <- which(degree(x) == 0)
  return (delete.vertices(x, isolates))
}

headers <- readLines("C:/dev/temp/testcorpus3/out/rescal/headers.txt")
incon <- readMM("C:/dev/temp/testcorpus3/out/rescal/incon.mtx")
outcon <- readMM("C:/dev/temp/testcorpus3/out/rescal/outcon.mtx")
needtype <- readMM("C:/dev/temp/testcorpus3/out/rescal/needtype.mtx")

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
g_incon <- deleteIsolates(g_incon)
g_outcon <- deleteIsolates(g_outcon)
g_union <- graph.union(g_incon, g_outcon)

vcount(g_incon)
vcount(g_outcon)
ecount(g_incon)
ecount(g_outcon)

V(g_incon)$color <- ifelse(V(g_incon)$type=="OFFER", "blue", ifelse(V(g_incon)$type=="WANT", "red", "green"))
V(g_union)$color <- ifelse(V(g_outcon)$type=="OFFER", "blue", ifelse(V(g_outcon)$type=="WANT", "red", "green"))
#V(g_outcon)$size <- degree(g_outcon)/10 + 3

layall <- layout.auto(g_union)
layall <- layout.norm(layall, xmin = -1, xmax = 1, ymin = -1, ymax = 1)
layin <- layall[which(is.element(V(g_incon)$name, V(g_union)$name)),]

plot(g_incon, layout=layin, edge.curved=TRUE)
plot(g_union, layout=layall, edge.curved=TRUE)

target <- identify(layall[,1], layall[,2])
V(g_outcon)[target]$need

