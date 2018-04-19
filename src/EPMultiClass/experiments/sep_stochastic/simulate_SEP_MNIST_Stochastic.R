source("../../ep_code/sep_stochastic.R")

set.seed(0)

load("../../data/mnist.dat")

mtr <- apply(data$Xtr, 2, mean)
str <- apply(data$Xtr, 2, sd)
str[ str == 0 ] <- 1

data$Xtr <- (data$Xtr - matrix(mtr, nrow(data$Xtr), ncol(data$Xtr), byrow = TRUE)) / matrix(str, nrow(data$Xtr), ncol(data$Xtr), byrow = TRUE)
data$Xts <- (data$Xts - matrix(mtr, nrow(data$Xts), ncol(data$Xts), byrow = TRUE)) / matrix(str, nrow(data$Xts), ncol(data$Xts), byrow = TRUE)

s <- sample(1 : nrow(data$Xtr))[ 1 : 6e4 ]

data$Xtr <- data$Xtr[ s , ]
data$Ytr <- 2 * as.integer(is.element(data$Ytr[ s ], c(0, 2, 4, 6, 8))) - 1
data$Yts <- 2 * as.integer(is.element(data$Yts, c(0, 2, 4, 6, 8))) - 1

gc()

spgpc <- epGPCInternal(data$Xtr, data$Ytr, n_pseudo_inputs = 200, data$Xts[ 1 : 1e4, ], data$Yts[ 1 : 1e4 ])


