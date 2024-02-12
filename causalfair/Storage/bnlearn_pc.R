library(bnlearn)

data <- read.csv("Storage/data.csv")

data <- sapply( data, as.numeric ) #factor for discrete
data <- as.data.frame(data)
obj_hc = pc.stable(data)

arc_mat = as.data.frame(obj_hc[["arcs"]])
write.csv(arc_mat, "Storage/edge_df.csv", row.names = FALSE)

