source("../../vi_code/svi_multicore.R")

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

REPORT <- FALSE
CONT <- 1

t1 <- as.double(Sys.time())
time <- system.time(spgpc <- fit_SGPC(data$Xtr, data$Ytr, n_pseudo_inputs = 200, nNodes = 2))
t2 <- as.double(Sys.time())

# We make predictions

prediction <- predict_SGPC(data$Xts, spgpc)

# We evaluate the classification error

error <- mean(data$Yts != sign(prediction - 0.5))

# We evaluate the test log-likelihood
	
ll <- mean(log(prediction * (data$Yts == 1) + (1 - prediction) * (data$Yts == -1)))

write.table(error, paste("results/test_error.txt", sep = ""), col.names = F, row.names = F, append = T)

write.table(ll, paste("results/test_ll.txt", sep = ""), col.names = F, row.names = F, append = T)

write.table(t(as.matrix(time)), paste("results/time.txt", sep = ""), col.names = F, row.names = F, append = T)

write.table(t2 - t1, paste("results/time_diff.txt", sep = ""), col.names = F, row.names = F, append = T)

