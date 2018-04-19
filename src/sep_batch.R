library(doMC)

# Global variable that indicates the current repetition

rep <- 1
t0 <- NULL

eps <- NULL
sign <- NULL

##
# Function which computes the cholesky decomposition of the inverse
# of a particular matrix.
#
# @param	M	m x m positive definite matrix.
#
# @return	L	m x m upper triangular matrix such that
#			M^-1 = L %*% t(L)
#

cholInverse <- function(M) { rot180(forwardsolve(t(chol(rot180(M))), diag(nrow(M)))) }

##
# Function which rotates a matrix 180 degreees.
#

rot180 <- function(M) { matrix(rev(as.double(M)), nrow(M), ncol(M)) }

##
# This function computes the covariance matrix for the GP
#

kernel <- function(X, l, sigma0, sigma) {
	X <- X / matrix(sqrt(l), nrow(X), ncol(X), byrow = TRUE)
	distance <- as.matrix(dist(X))^2
	sigma * exp(-0.5 * distance) + diag(sigma0, nrow(X)) + diag(rep(1e-10, nrow(X)))
}

##
# Function which computes the kernel matrix between the observed data and the test data
#

kernel_nm <- function(X, Xnew, l, sigma) {

	X <- X / matrix(sqrt(l), nrow(X), ncol(X), byrow = TRUE)	
	Xnew <- Xnew / matrix(sqrt(l), nrow(Xnew), ncol(Xnew), byrow = TRUE)
	n <- nrow(X)
	m <- nrow(Xnew)
	Q <- matrix(apply(X^2, 1, sum), n, m)
	Qbar <- matrix(apply(Xnew^2, 1, sum), n, m, byrow = T)
	distance <- Qbar + Q - 2 * X %*% t(Xnew)
	sigma * exp(-0.5 * distance)
}

##
# Function which computes the diagonal of the kernel matrix for the data
# points.
#
# @param	X 		n x d matrix with the n data points.
# @param	sigma		scalar with the amplitude of the GP.
# @param	sigma0		scalar with the noise level in the GP.
#
# @return	diagKnn		n-dimensional vector with the diagonal of the
#				kernel matrix for the data points.
#

computeDiagKernel <- function(X, sigma, sigma0) { rep(sigma, nrow(X)) + 1e-10 + sigma0 }

##
# Function that initializes the struture with the problem information.
#
# @param	X	n x d matrix with the data points.
# @param	Xbar	m x d matrix with the pseudo inputs.
# @param	sigma	scalar with the log-amplitude of the GP.
# @param	sigma0	scalar with the log-noise level in the GP.
# @param	l	d-dimensional vector with the log-lengthscales.
# 
# @return	gFITCinfo	List with the problem information 
#

initialize_kernel_FITC <- function(Y, X, Xbar, sigma, sigma0, l) {

	# We initialize the structure with the data and the kernel
	# hyper-parameters

	gFITCinfo <- list()
	gFITCinfo$X <- X
	gFITCinfo$Y <- Y
	gFITCinfo$Xbar <- Xbar
	gFITCinfo$m <- nrow(Xbar)
	gFITCinfo$d <- ncol(Xbar)
	gFITCinfo$n <- nrow(X)
	gFITCinfo$sigma <- sigma
	gFITCinfo$sigma0 <- sigma0
	gFITCinfo$l <- l

	# We compute the kernel matrices

	gFITCinfo$Kmm <- kernel(Xbar, gFITCinfo$l, gFITCinfo$sigma0, gFITCinfo$sigma)
	gFITCinfo$KmmInv <- chol2inv(chol(gFITCinfo$Kmm))
	gFITCinfo$Knm <- kernel_nm(X, Xbar, gFITCinfo$l, gFITCinfo$sigma)
	gFITCinfo$P <- gFITCinfo$Knm
	gFITCinfo$R <- cholInverse(gFITCinfo$Kmm)
	gFITCinfo$PRt <- gFITCinfo$P %*% t(gFITCinfo$R)
	gFITCinfo$PRtR <- gFITCinfo$PRt %*% gFITCinfo$R

	# We compute the diagonal matrices

	gFITCinfo$diagKnn <- computeDiagKernel(X, gFITCinfo$sigma, gFITCinfo$sigma0)
	gFITCinfo$diagPRtRPt <- gFITCinfo$PRt^2 %*% rep(1, gFITCinfo$m)
	gFITCinfo$D <- gFITCinfo$diagKnn - gFITCinfo$diagPRtRPt

	gFITCinfo
}

##
# Function that computes the marginal means and variances of the product
# of the FITC prior and a multivariate Gaussian density with diagonal
# correlation matrix.
#
# @param	gFITCinfo	List with the problem information
#				(see initializegFITCinfo).
# @param	f1Hat		list with n-dimensional vector with the inverse
#				variances times the mean of the Gaussian and the invese variances
#				approximation to the likelihood factor.
#
# @return	ret		A list with the marginal means and variances.
#

computeTitledDistribution <- function(nodeInfo) {

	Vinv <- nodeInfo$gFITCinfo$KmmInv
	mNew <- rep(0, nodeInfo$gFITCinfo$m)

	Vinv <- Vinv + nodeInfo$approx$eta2
	mNew <- mNew + nodeInfo$approx$eta1

	L <- chol(Vinv)
	vNew <- chol2inv(L)
	mNew <- vNew %*% mNew

	list(mNew = mNew, vNew = vNew, log_det_vNew = - 2 * sum(log(diag(L))))
}

##
# Function that computes classes and probabilities of the labels of test data
#
# ret is the list returned by EP
#

predict <- function(Xtest, ret) {

	# We compute the FITC prediction

	posterior <- computeTitledDistribution(ret$nodeInfo)

	P_new <- kernel_nm(Xtest, ret$nodeInfo$gFITCinfo$Xbar, ret$nodeInfo$gFITCinfo$l, ret$nodeInfo$gFITCinfo$sigma)
	diagKnn_new <- computeDiagKernel(Xtest, ret$nodeInfo$gFITCinfo$sigma, ret$nodeInfo$gFITCinfo$sigma0)

	PRtR_new <- P_new %*% t(ret$nodeInfo$gFITCinfo$R) %*% ret$nodeInfo$gFITCinfo$R
	
	gamma <- colSums(t(PRtR_new) * (posterior$vNew %*% t(PRtR_new)))

	z <- diagKnn_new - rowSums(PRtR_new * P_new) + gamma 
	theta <- PRtR_new %*% posterior$mNew

	mPrediction <- theta
	vPrediction <- z + 1
	
	pnorm(mPrediction / sqrt(vPrediction))
}

##
# This function computes the gradients of the ML approximation provided by EP once it has converged
#

computeGradsPrior <- function(l, sigma0, sigma, nodeInfo) {

	# We compute some matrices that are needed for the gradient 

	posterior <- computeTitledDistribution(nodeInfo)

	mNew <- posterior$mNew
	vNew <- posterior$vNew

	gFITCinfo <- nodeInfo$gFITCinfo

	Kinv <- gFITCinfo$KmmInv
	K <- gFITCinfo$Kmm
	M <- Kinv - Kinv %*% (vNew %*% Kinv) - ((Kinv %*% mNew) %*% (t(mNew) %*% Kinv))

	# We compute the derivatives of the kernel with respect to log_sigma

	dKmm_dlog_sigma0 <- diag(gFITCinfo$m) * sigma0

	gr_log_sigma0 <- -0.5 * sum(M * dKmm_dlog_sigma0)

	# We compute the derivatives of the kernel with respect to log_sigma0

	dKmm_dlog_sigma <- gFITCinfo$Kmm - diag(rep(gFITCinfo$sigma0, gFITCinfo$m)) - diag(1e-10, gFITCinfo$m)

	gr_log_sigma <- -0.5 * sum(M * dKmm_dlog_sigma)

	# We compute the derivatives of the kernel with respect to l

#	gr_log_l <- rep(0, length(l))

#	for (i in 1 : length(l)) {
#		distance <- as.matrix(dist(gFITCinfo$Xbar[, i, drop = FALSE ]))^2
#		dKmm_dlog_l <- gFITCinfo$Kmm * 0.5 * distance / l[ i ]
#		gr_log_l[ i ] <- - 0.5 * sum(M * dKmm_dlog_l)
#	}

	# The distance is v^2 1^T - 2 v v^T + 1^T v^2

	Ml <- 0.5 * M * gFITCinfo$Kmm
	Xl <-  (gFITCinfo$Xbar / matrix(sqrt(l), nrow(gFITCinfo$Xbar), ncol(gFITCinfo$Xbar), byrow = TRUE))
	gr_log_l <- - 2 * 0.5 * colSums(Ml %*% Xl^2) + 2 * 0.5 * colSums(Xl * (Ml %*% Xl))

#	gr_xbar <- matrix(0, gFITCinfo$m, length(l)) 
#
#	for (i in 1 : length(l)) {
#
#		distance <- (matrix(gFITCinfo$Xbar[ , i ], gFITCinfo$m, gFITCinfo$m) - matrix(gFITCinfo$Xbar[ , i ],
#			gFITCinfo$m, gFITCinfo$m, byrow = T))
#
#		dKmm_dXbar <- - (gFITCinfo$Kmm - diag(gFITCinfo$sigma0, gFITCinfo$m) - diag(1e-10, gFITCinfo$m)) * distance / gFITCinfo$l[ i ]
#
#		gr_xbar[ ,i ] <- - 0.5 * rowSums(M * dKmm_dXbar) + 0.5 * colSums(M * dKmm_dXbar)
#	}

	# The derivative of the distance is dk 1^T - 1^T dk where k is the number of the pseudoinput
	# the distance is vi 1^T - 1^T vi

	Xbar <- (gFITCinfo$Xbar / matrix(l, nrow(gFITCinfo$Xbar), ncol(gFITCinfo$Xbar), byrow = TRUE))
	Mbar <- t(M) * (- (gFITCinfo$Kmm - diag(gFITCinfo$sigma0, gFITCinfo$m) - diag(1e-10, gFITCinfo$m)))
	gr_xbar <- - 0.5 * 2 * (Xbar * matrix(rep(1, gFITCinfo$m) %*% Mbar, gFITCinfo$m, length(l)) - Mbar %*% Xbar)

	list(gr_log_l = gr_log_l, gr_log_sigma0 = gr_log_sigma0, gr_log_sigma = gr_log_sigma, gr_xbar = gr_xbar)
}

##
# This function computes the gradients of the ML approximation provided by EP once it has converged
#
computeGradsLikelihood <- function(l, sigma0, sigma, nodeInfo) {


	posterior <- computeTitledDistribution(nodeInfo)

	mNew <- posterior$mNew
	vNew <- posterior$vNew

	gr_l <- computeGradsLikelihoodNode(vNew, mNew, nodeInfo$gFITCinfo, l, sigma0, sigma, nodeInfo$f1Hat)

#	gr_l <- list()
#	for (n in 1 : nNodes) {
#		gr_l[[ n ]] <- computeGradsLikelihoodNode(vNew, mNew, nodeInfo[[ n ]]$gFITCinfo, l, sigma0, sigma, nodeInfo[[ n ]]$f1Hat)
#	}

	gr_l
}

computeGradsLikelihoodNode <- function(vNew, mNew, gFITCinfo, l, sigma0, sigma, f1Hat) {

	# Loop through the data

	log_evidence <- 0
	n <- gFITCinfo$n
	m <- gFITCinfo$m

	# We precompute some derivatives

	dKnn_dlog_sigma0 <- rep(sigma0, gFITCinfo$n)
	dKmm_dlog_sigma0 <- diag(gFITCinfo$m) * sigma0
	dP_dlog_sigma0 <- matrix(0, gFITCinfo$n, gFITCinfo$m)

	dKnn_dlog_sigma <- rep(sigma, gFITCinfo$n)
	dKmm_dlog_sigma <- gFITCinfo$Kmm - diag(rep(gFITCinfo$sigma0, gFITCinfo$m)) - diag(1e-10, gFITCinfo$m)
	dP_dlog_sigma <- gFITCinfo$P 

	# We compute dlog_evidece_dPRtR and dlog_evidece_dP and dlog_evidece_dKnn

	dlog_evidece_dPRtR <- matrix(0, n, m)
	dlog_evidece_dP <- matrix(0, n, m)
	dlog_evidece_dKnn <- rep(0, n)

	PRtRvNew <- gFITCinfo$PRtR %*% vNew
	PRtRvNewRtRPt <- rowSums(PRtRvNew * gFITCinfo$PRtR)
	C1 <- (f1Hat$eta2^-1 - PRtRvNewRtRPt)^-1
	PRtRvOldRtRPt <- PRtRvNewRtRPt + PRtRvNewRtRPt^2 * C1
	PRtRvOld <- PRtRvNew + matrix(PRtRvNewRtRPt * C1, n, m) * PRtRvNew 
	PRtRmNew <- gFITCinfo$PRtR %*% mNew
	C2 <- f1Hat$eta2 * PRtRmNew - f1Hat$eta1
	PRtRmOld <- PRtRmNew + PRtRvOldRtRPt * C2
	mOldMatrix <- matrix(mNew, n, m, byrow = TRUE) + (PRtRvNew + PRtRvNew * matrix(PRtRvNewRtRPt * C1, n, m)) * matrix(C2, n, m)
		
	z <- gFITCinfo$D + PRtRvOldRtRPt + 1

	theta <- PRtRmOld
	logZ <-  pnorm(gFITCinfo$Y * theta / sqrt(z), log.p = TRUE) 
	alpha <-  gFITCinfo$Y / sqrt(z) * exp(dnorm(gFITCinfo$Y * theta / sqrt(z), 0, 1, log = TRUE) - 
		pnorm(gFITCinfo$Y * theta / sqrt(z), log.p = TRUE))

	dlog_evidece_dPRtR <- matrix(exp(-logZ + dnorm(gFITCinfo$Y * theta / sqrt(z), log = TRUE)) * gFITCinfo$Y, n, m) *
		(mOldMatrix / matrix(sqrt(z), n, m) - matrix(0.5 * theta * 1 / z^(3/2), n, m) * (-gFITCinfo$P + 2 * PRtRvOld)) 

	dlog_evidece_dP <- matrix(exp(-logZ + dnorm(gFITCinfo$Y * theta / sqrt(z), log = TRUE)) * gFITCinfo$Y, n, m) * 
		(matrix(0.5 * theta * 1 / z^(3 / 2), n, m) * gFITCinfo$PRtR) 
	dlog_evidece_dKnn <- exp(-logZ + dnorm(gFITCinfo$Y * theta / sqrt(z), log = TRUE)) * 
		gFITCinfo$Y * (- 0.5 * theta * 1 / z^(3/2)) 

	# We now compute the actual gradients

	M1 <- gFITCinfo$KmmInv %*% t(dlog_evidece_dPRtR)
	M2 <- - M1 %*% gFITCinfo$PRtR
	M3 <- dlog_evidece_dP

	gr_log_sigma <- sum(t(M1) * dP_dlog_sigma) + sum(t(M2) * dKmm_dlog_sigma) + sum(t(dlog_evidece_dKnn) * dKnn_dlog_sigma) + 
		sum(M3 * dP_dlog_sigma) 

	gr_log_sigma0 <- sum(t(M1) * dP_dlog_sigma0) + sum(t(M2) * dKmm_dlog_sigma0) + sum(t(dlog_evidece_dKnn) * dKnn_dlog_sigma0) +
		sum(M3 * dP_dlog_sigma0) 

#	gr_log_l <- rep(0, length(l)) 
#
#	for (i in 1 : length(l)) {
#
#		dKnn_dlog_l <- rep(0, gFITCinfo$n)
#		distance <- as.matrix(dist(gFITCinfo$Xbar[, i, drop = FALSE ]))^2
#		dKmm_dlog_l <- gFITCinfo$Kmm * 0.5 * distance / l[ i ]
#
#		Q <- matrix(gFITCinfo$X[ , i ]^2, gFITCinfo$n, gFITCinfo$m)
#               Qbar <- matrix(gFITCinfo$Xbar[ , i ]^2, gFITCinfo$n, gFITCinfo$m, byrow = T)
#               distance <- Qbar + Q - 2 * gFITCinfo$X[ , i ] %*% t(gFITCinfo$Xbar[ , i ])
#                dP_dlog_l <- gFITCinfo$P * 0.5 * distance / l[ i ]
#
#		gr_log_l[ i ] <- sum(t(M1) * dP_dlog_l) + sum(t(M2) * dKmm_dlog_l) + sum(t(dlog_evidece_dKnn) * dKnn_dlog_l) +
#			 sum(M3 * dP_dlog_l)
#	}

	Ml <- 0.5 * (t(M1) + M3) * gFITCinfo$P
	Xbarl <-  (gFITCinfo$Xbar / matrix(sqrt(l), nrow(gFITCinfo$Xbar), ncol(gFITCinfo$Xbar), byrow = TRUE))
	Xl <-  (gFITCinfo$X / matrix(sqrt(l), nrow(gFITCinfo$X), ncol(gFITCinfo$X), byrow = TRUE))
	Ml2 <- t(M2) * gFITCinfo$Kmm * 0.5
	gr_log_l <- colSums(t(Ml) %*% Xl^2) - 2  * colSums(Xl * (Ml %*% Xbarl)) +  colSums(Ml %*% Xbarl^2) + 
		colSums(t(Ml2) %*% Xbarl^2) - 2 * colSums(Xbarl * (Ml2 %*% Xbarl)) + colSums(Ml2 %*% Xbarl^2)

#	gr_xbar <- matrix(0, gFITCinfo$m, length(l)) 
#
#	for (i in 1 : length(l)) {
#
#		distance <- matrix(gFITCinfo$X[ , i ], gFITCinfo$n, gFITCinfo$m) - matrix(gFITCinfo$Xbar[ , i ], 
#			gFITCinfo$n, gFITCinfo$m, byrow = T)
#		dP_dXbar <- gFITCinfo$P * distance / gFITCinfo$l[ i ]
#
#		distance <- (matrix(gFITCinfo$Xbar[ , i ], gFITCinfo$m, gFITCinfo$m) - matrix(gFITCinfo$Xbar[ , i ],
#			gFITCinfo$m, gFITCinfo$m, byrow = T))
#		dKmm_dXbar <- (gFITCinfo$Kmm - diag(gFITCinfo$sigma0, gFITCinfo$m) - diag(1e-10, gFITCinfo$m)) * - distance / gFITCinfo$l[ i ]
#
#		dKnn_dXbar <- rep(0, gFITCinfo$n)
#
#		gr_xbar[ ,i ] <- colSums(t(M1) * dP_dXbar) - colSums(t(M2) * dKmm_dXbar) + colSums(t(dKmm_dXbar) * M2) + 
#			rep(sum(t(dlog_evidece_dKnn) * dKnn_dXbar), gFITCinfo$m) + colSums(M3 * dP_dXbar)
#		
#	}

	Xbar <- (gFITCinfo$Xbar / matrix(l, nrow(gFITCinfo$Xbar), ncol(gFITCinfo$Xbar), byrow = TRUE))
	X <- (gFITCinfo$X / matrix(l, nrow(gFITCinfo$X), ncol(gFITCinfo$X), byrow = TRUE))
	Mbar <- t(M2) * - (gFITCinfo$Kmm - diag(gFITCinfo$sigma0, gFITCinfo$m) - diag(1e-10, gFITCinfo$m)) 
	Mbar2 <- (t(M1) + M3) * gFITCinfo$P
	gr_xbar <- (Xbar * matrix(rep(1, gFITCinfo$m) %*% Mbar, gFITCinfo$m, length(l)) - t(Mbar) %*% Xbar) +
			(Xbar * matrix(rep(1, gFITCinfo$m) %*% t(Mbar), gFITCinfo$m, length(l)) - Mbar %*% Xbar) + 
			(t(Mbar2) %*% X) - ((Xbar * matrix(rep(1, gFITCinfo$n) %*% Mbar2, gFITCinfo$m, length(l))))

	list(gr_log_l = gr_log_l, gr_log_sigma0 = gr_log_sigma0, gr_log_sigma = gr_log_sigma, gr_xbar = gr_xbar)
}

##
# Function that estimates the initial lengthscale value

estimateL <- function (X) {

	D <- as.matrix(dist(X))
	median(D[ upper.tri(D) ])
}

##
# This function runs EP until convergence
#

epGPCInternal <- function(X, Y, n_pseudo_inputs, eps = 1e-2, iterations = 250, start = NULL, damping = 0.5, kmeans = FALSE) {

        t0 <<- proc.time()

        log_sigma <- 0
        log_sigma0 <- log(1e-3)

	if (kmeans == FALSE) 
	        Xbar <- X[ sample(1 : nrow(X), n_pseudo_inputs), , drop = F ]
	else
		Xbar <- kmeans(cbind(Y, X), n_pseudo_inputs)$centers[, -1 ] 
		
        log_l <- rep(log(estimateL(Xbar)), ncol(Xbar))

	l <- exp(log_l)
	sigma0 <- exp(log_sigma0)
	sigma <- exp(log_sigma)

	m <- nrow(Xbar)
	n <- nrow(X)

	eps <<- list(sigma = eps, sigma0 = eps, l = rep(eps, length(log_l)), xbar = matrix(eps, nrow(Xbar), ncol(Xbar)))
	rm(eps)
	sign <<- list(sigma = NULL, sigma0 = NULL, l = rep(NULL, length(log_l)), xbar = matrix(0, nrow(Xbar), ncol(Xbar)))

	# We split the data for each node

	nodeInfo <- Ynode <- qApprox <- gFITCinfo <- f1Hat <- Xnode <- list()

	Xnode <- X
	Ynode <- Y
	f1Hat <- list(eta1 = rep(0, nrow(Xnode)), eta2 = rep(0, nrow(Xnode)), eta_vectors = matrix(0, nrow(Xnode), m))
	gFITCinfo <- initialize_kernel_FITC(Ynode, Xnode, Xbar, sigma, sigma0, l)
	approx <- list(eta1 = rep(0, m), eta2 = matrix(0, m, m))
	nodeInfo <- list(f1Hat = f1Hat, gFITCinfo = gFITCinfo, approx = approx)

	if (! is.null(start)) {
		nodeInfo$approx <- start$nodeInfo$approx
		nodeInfo$f1Hat <- start$nodeInfo$f1Hat
	}

	# We check for an initial solution
	

	# Main loop of EP

	i <- 1
	convergence <- FALSE
	nodeInfoOld <- NULL

	while (i < iterations) {

		update_correct <- FALSE
		second_update <- fail <- FALSE
		damping_inner <- damping

		while(update_correct != TRUE) {


			error <- FALSE

			posterior <- computeTitledDistribution(nodeInfo)

			mNew <- posterior$mNew
			vNew <- posterior$vNew
	
			# We do the parallel updates

			nodeInfoNew <- nodeInfo
	
			PRtRvNewRtRPt <- rowSums((nodeInfoNew$gFITCinfo$PRtR %*% vNew) * nodeInfoNew$gFITCinfo$PRtR)
			C1 <- (nodeInfoNew$f1Hat$eta2^-1 - PRtRvNewRtRPt)^-1
			PRtRvOldRtRPt <- PRtRvNewRtRPt + PRtRvNewRtRPt^2 * C1
			PRtRmNew <- nodeInfoNew$gFITCinfo$PRtR %*% mNew
			C2 <- nodeInfoNew$f1Hat$eta2 * PRtRmNew - nodeInfoNew$f1Hat$eta1
			PRtRmOld <- PRtRmNew + PRtRvOldRtRPt * C2
				
			z <- nodeInfoNew$gFITCinfo$D + PRtRvOldRtRPt + 1

			theta <- PRtRmOld
            
			alpha <- nodeInfoNew$gFITCinfo$Y / sqrt(z) * exp(dnorm(nodeInfoNew$gFITCinfo$Y * 
				theta / sqrt(z), 0, 1, log = TRUE) - pnorm(nodeInfoNew$gFITCinfo$Y * theta / sqrt(z), log.p = TRUE))
	
			eta2new <- (alpha^2 + alpha * theta / z) * (1 - (alpha^2 + alpha * theta / z) * PRtRvOldRtRPt)^-1
			eta1new <- eta2new * theta + alpha + alpha * PRtRvOldRtRPt * eta2new

			nodeInfoNew$f1Hat$eta1 <- (1 - damping_inner) *  nodeInfoNew$f1Hat$eta1  + damping_inner * eta1new
			nodeInfoNew$f1Hat$eta2 <- (1 - damping_inner) *  nodeInfoNew$f1Hat$eta2  + damping_inner * eta2new
			nodeInfoNew$f1Hat$eta_vectors <-  nodeInfoNew$gFITCinfo$PRtR

			nodeInfoNew$approx$eta2 <- t(nodeInfoNew$f1Hat$eta_vectors) %*% (matrix(nodeInfoNew$f1Hat$eta2, nodeInfoNew$gFITCinfo$n, 
				nodeInfoNew$gFITCinfo$m) * nodeInfoNew$f1Hat$eta_vectors) 

			nodeInfoNew$approx$eta1 <- colSums(matrix(nodeInfoNew$f1Hat$eta1, nodeInfoNew$gFITCinfo$n, 
				nodeInfoNew$gFITCinfo$m) * nodeInfoNew$f1Hat$eta_vectors)

			# We update the marginal likelihood
	
			nodeInfoNew <- optimize(Y, X, nodeInfoNew)$nodeInfo

			evidence <- computeEvidence(nodeInfoNew)

			if (is.nan(evidence))  {
				error <- TRUE
			} 

			if (error == FALSE) {
				if (fail == TRUE && second_update == FALSE) {
					nodeInfo  <- nodeInfoNew
					second_update <- TRUE
				} else 
					update_correct <- TRUE
			} else {

				cat("Reducing damping factor to guarantee EP update and eps! Damping:", damping_inner, "\n")

				nodeInfo  <- nodeInfoOld
				damping_inner <- damping_inner * 0.5
				fail <- TRUE
				second_update <- FALSE

				eps$sigma <<- eps$sigma * 0.5
				eps$sigma0 <<- eps$sigma0 * 0.5
				eps$l <<- eps$l * 0.5
				eps$xbar <<- eps$xbar * 0.5
			}
		}

		nodeInfoOld <- nodeInfo
		nodeInfo <- nodeInfoNew

		# We check for convergence

		change <- max(abs(nodeInfoOld$approx$eta1 - nodeInfo$approx$eta1))
		change <- max(change, abs(nodeInfoOld$approx$eta2 - nodeInfo$approx$eta2))

#		if (change < 1e-3)
#			convergence <- T
	
		cat("\tIteration",  i, change, "Evidence:", evidence, "\n")

		if (REPORT == TRUE) {

			ret <- 	list(nodeInfo = nodeInfo)

			t_before <- proc.time()
			prediction <- predict(Xtest, ret)
			ee <- mean(sign(prediction - 0.5) != ytest)
			ll <- mean(log(prediction * (ytest == 1) + (1 - prediction) * (ytest == -1)))
			t_after <- proc.time()

			t0 <- t0 + (t_after - t_before)

			write.table(t(c(ee, ll, proc.time() - t0)), 
				file = paste("./results/time_outter_", CONT, ".txt", sep = ""), row.names = F, col.names = F, append = TRUE)
		}

		# Annealed damping scheme

#		damping <- damping * 0.99

		i <- i + 1

#		gc()
	}

	l <- nodeInfo$gFITCinfo$l
	sigma <- nodeInfo$gFITCinfo$sigma
	sigma0 <- nodeInfo$gFITCinfo$sigma0
	Xbar <- nodeInfo$gFITCinfo$Xbar

	# We compute the evidence and its gradient
	
	logZ <- computeEvidence(nodeInfo)
	grad_prior <- computeGradsPrior(l, sigma0, sigma, nodeInfo)
	grad_likelihood <- computeGradsLikelihood(l, sigma0, sigma, nodeInfo)

	grad <- grad_prior
	grad$gr_log_l <- grad$gr_log_l + grad_likelihood$gr_log_l
	grad$gr_log_sigma <- grad$gr_log_sigma + grad_likelihood$gr_log_sigma
	grad$gr_log_sigma0 <- grad$gr_log_sigma0 + grad_likelihood$gr_log_sigma0
	grad$gr_xbar <- grad$gr_xbar + grad_likelihood$gr_xbar

	list(logZ = logZ, grad = grad, l = l, sigma0 = sigma0, sigma = sigma, X = X, Y = Y, nodeInfo = nodeInfo)
}

###
# Function which computes the EP approximation of the log evidence.
#
# @param	f1Hat		The approximation for the first factor.
# @param	gFITCinfo	The list with the problem information.
# @param	Y		The class labels.
#
# @return	logZ		The log evidence.
#

computeEvidence <- function(nodeInfo) {

	Vinv <- nodeInfo$gFITCinfo$KmmInv
	mNew <- rep(0, nodeInfo$gFITCinfo$m)

	Vinv <- Vinv + nodeInfo$approx$eta2
	mNew <- mNew + nodeInfo$approx$eta1

	L <- chol(Vinv)
	vNew <- chol2inv(L)
	mNew <- vNew %*% mNew
	log_det_vNew <- - 2 * sum(log(diag(L)))

	# Loop through the data

	log_evidence <- 0

	n <- nodeInfo$gFITCinfo$n
	m <- nodeInfo$gFITCinfo$m

	PRtRvNewRtRPt <- rowSums((nodeInfo$gFITCinfo$PRtR %*% vNew) * nodeInfo$gFITCinfo$PRtR)
	C1 <- (nodeInfo$f1Hat$eta2^-1 - PRtRvNewRtRPt)^-1
	PRtRvOldRtRPt <- PRtRvNewRtRPt + PRtRvNewRtRPt^2 * C1
	PRtRmNew <- nodeInfo$gFITCinfo$PRtR %*% mNew
	C2 <- nodeInfo$f1Hat$eta2 * PRtRmNew - nodeInfo$f1Hat$eta1
	PRtRmOld <- PRtRmNew + PRtRvOldRtRPt * C2
	mOldVinvmOld <- sum(t(mNew) %*% Vinv %*% mNew) + 2 * (PRtRmNew + PRtRmNew * PRtRvNewRtRPt * C1) * C2 + 
		(PRtRvNewRtRPt + PRtRvNewRtRPt^2 * C1) * C2^2 + (PRtRvNewRtRPt^2 * C1 + PRtRvNewRtRPt^3 * C1^2) * C2^2
			
	z <- nodeInfo$gFITCinfo$D + PRtRvOldRtRPt + 1

	theta <- PRtRmOld
	logZ <-  pnorm(nodeInfo$gFITCinfo$Y * theta / sqrt(z), log.p = TRUE) + m / 2 * log(2 * pi) + 
		0.5 * mOldVinvmOld - 0.5 * nodeInfo$f1Hat$eta2 * PRtRmOld^2 + 
		0.5 * log_det_vNew - 0.5 * log(1 - nodeInfo$f1Hat$eta2 * PRtRvNewRtRPt)

	logZtilde <-  m / 2 * log(2 * pi) + 0.5 * sum(mNew * (Vinv %*% mNew)) + 0.5 * log_det_vNew
	
	log_evidence <- log_evidence + sum(logZ - logZtilde)

	m <- nodeInfo$gFITCinfo$m
	
	log_evidence <- log_evidence + m / 2 * log(2 * pi) + 0.5 * sum(mNew * (Vinv %*% mNew)) + 0.5 * log_det_vNew
	log_evidence <- log_evidence - m / 2 * log(2 * pi) - 0.5 * sum(- 2 * log(diag(nodeInfo$gFITCinfo$R)))

	log_evidence
}


##
# This function updates the kernel hyper-parameters
#

optimize <- function(Y, X, nodeInfo) {

	l <- nodeInfo$gFITCinfo$l
	sigma <- nodeInfo$gFITCinfo$sigma
	sigma0 <- nodeInfo$gFITCinfo$sigma0
	Xbar <- nodeInfo$gFITCinfo$Xbar

	# We compute the gradient

	grad_prior <- computeGradsPrior(l, sigma0, sigma, nodeInfo)
	grad_likelihood <- computeGradsLikelihood(l, sigma0, sigma, nodeInfo)

	grad <- grad_prior
	grad$gr_log_l <- grad$gr_log_l + grad_likelihood$gr_log_l
	grad$gr_log_sigma <- grad$gr_log_sigma + grad_likelihood$gr_log_sigma
	grad$gr_log_sigma0 <- grad$gr_log_sigma0 + grad_likelihood$gr_log_sigma0
	grad$gr_xbar <- grad$gr_xbar + grad_likelihood$gr_xbar

	if (! is.null(sign$sigma)) {

                eps$sigma[ sign$sigma == sign(grad$gr_log_sigma) ] <<- eps$sigma[ sign$sigma == sign(grad$gr_log_sigma) ] * 1.02
                eps$sigma[ sign$sigma != sign(grad$gr_log_sigma) ] <<- eps$sigma[ sign$sigma != sign(grad$gr_log_sigma) ] * 0.5

                eps$sigma0[ sign$sigma0 == sign(grad$gr_log_sigma0) ] <<- eps$sigma0[ sign$sigma0 == sign(grad$gr_log_sigma0) ] * 1.02
                eps$sigma0[ sign$sigma0 != sign(grad$gr_log_sigma0) ] <<- eps$sigma0[ sign$sigma0 != sign(grad$gr_log_sigma0) ] * 0.5

                eps$l[ sign$l == sign(sum(grad$gr_log_l)) ] <<- eps$l[ sign$l == sign(sum(grad$gr_log_l)) ] * 1.02
                eps$l[ sign$l != sign(sum(grad$gr_log_l)) ] <<- eps$l[ sign$l != sign(sum(grad$gr_log_l)) ] * 0.5

                eps$xbar[ sign$xbar == sign(grad$gr_xbar) ] <<- eps$xbar[ sign$xbar == sign(grad$gr_xbar) ] * 1.02
                eps$xbar[ sign$xbar != sign(grad$gr_xbar) ] <<- eps$xbar[ sign$xbar != sign(grad$gr_xbar) ] * 0.5
	}

	sign$sigma <<- sign(grad$gr_log_sigma)
	sign$sigma0 <<- sign(grad$gr_log_sigma0)
	sign$l <<- sign(sum(grad$gr_log_l))
	sign$xbar <<- sign(grad$gr_xbar)

	l <- exp(log(l) + eps$l * grad$gr_log_l)
	sigma0 <- exp(log(sigma0) + eps$sigma0 * grad$gr_log_sigma0)
	sigma <- exp(log(sigma) + eps$sigma * grad$gr_log_sigma)
	Xbar <- Xbar + eps$xbar * grad$gr_xbar

	Xnode <- X
	Ynode <- Y
	nodeInfo$gFITCinfo <- initialize_kernel_FITC(Ynode, Xnode, Xbar, sigma, sigma0, l)

	list(nodeInfo = nodeInfo, Xbar = Xbar, l = l, sigma0 = sigma0, sigma = sigma)
}

