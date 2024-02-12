library(bnlearn)

data <- read.csv("Storage/data.csv")
black_list_df <- read.csv("Storage/black_list.csv")
black_list = as.matrix(black_list_df)

data <- sapply( data, as.numeric ) #factor for discrete
data <- as.data.frame(data)
obj_hc = hc(data, blacklist = black_list)

arc_mat = as.data.frame(obj_hc[["arcs"]])
write.csv(arc_mat, "Storage/edge_df.csv", row.names = FALSE)

