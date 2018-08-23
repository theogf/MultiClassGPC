using Plots
using RCall
using Dates
using PyCall
using Distributions
@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp
N_samples = 1000
N_dim = 2
N_class = 3
N_test = 50
minx=-5.0
maxx=5.0
noise = 1.0
truthknown = false
doMCCompare = false
dolikelihood = false
println("$(now()): Starting testing multiclass")

function latent(X)
    return sqrt.(X[:,1].^2+X[:,2].^2)
end
X = (rand(N_samples,N_dim)*(maxx-minx))+minx
trunc_d = Truncated(Normal(0,3),minx,maxx)
X = rand(trunc_d,N_samples,N_dim)
x_test = linspace(minx,maxx,N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
# X_test = rand(trunc_d,N_test^dim,dim)
y = min.(max.(1,floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))),N_class)
y_test =  min.(max.(1,floor.(Int64,latent(X_test))),N_class)
# y = y.-1; y_tes
# X,y = sk.make_classification(n_samples=N_samples,n_features=N_dim,n_classes=N_class,n_clusters_per_class=1,n_informative=N_dim,n_redundant=0)
# X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
#Test on the Iris dataet
# train = readdlm("data/iris-X")
# X = train[1:100,:]; X_test=train[101:end,:]
# test = readdlm("data/iris-y")
# y = test[1:100,:]; y_test=test[101:end,:]
# truthknown = false
#
# data = readdlm("data/Glass")
# # Dataset has been already randomized
# X = data[1:150,1:(end-1)]; y=data[1:150,end]
# X_test = data[151:end,1:(end-1)]; y_test=data[151:end,end]
#### Test on the mnist dataset
# X = readdlm("data/mnist_train")
# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/mnist_test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]
# println("$(now()): MNIST data loaded")
#
# ### Test on the artificial character dataset
# X = readdlm("data/artificial-characters-train")
# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/artificial-characters-test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]
# println("$(now()): Artificial Characters data loaded")

R"source('src/sepMGPC_stochastic.R')"
m=20
# ind_points = KMeansInducingPoints(X,m,10)
batchsize=40
maxiter=50
indpointsopt=true
 l = 1.0
 hyperparamopt = false
 target_result = "test/results/performances.txt"
 t_EP = @elapsed EPmodel = R"epMGPCInternal($X, $y, m = $m, n_minibatch = $batchsize, X_test=$X_test, Y_test=$y_test, max_iters = $maxiter,autotuning =TRUE)"
 R"y_EP <- predictMGPC($EPmodel,$X_test)"
y_EP = @rget y_EP
y_pred = map(x->parse(Int64,x),y_EP[:labels])
EP_score = 0
for (i,pred) in enumerate(y_pred)
    if pred == y_test[i]
        global EP_score += 1
    end
end
println("Accuracy is $(EP_score/length(y_test))")
