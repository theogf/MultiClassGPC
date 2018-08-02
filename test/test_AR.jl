using Plots
using MATLAB
using PyCall
using Distributions
@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp
N_samples = 1000
N_dim = 100
N_class = 10
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


dataset_name = "mnist";
# dataset_name = "bibtex";
# dataset_name = "eurlex";
# dataset_name = "omniglot";
# dataset_name = "amazoncat";

param = Dict{String,Any}()
data_path = "~/Competitors/augment-reduce/src/data";       # Replace with the path to your dataset
param["output_path"] = "~/MultiClassGPC/test/results";                   # Replace with the path to the output folder


param["method_name"] = "softmax_a&r";  # Sofmax A&R
# param["method_name"] = "probit_a&r";   # Multinomial probit A&R
# param["method_name"] = "logistic_a&r"; # Multinomial logistic A&R
# param["method_name"] = "ove";          # One-vs-each bound [Titsias, 2016]
# param["method_name"] = "botev";        # The approach by [Botev et al., 2017]
# param["method_name"] = "softmax";      # Exact softmax


nIterations = 1000; minibatchsize = 100;
param["flag_mexFile"] = 1;       # Use compiled mex files?
param["s2prior"] = Inf;          # Variance of the prior over the weights and intercepts (inf for maximum likelihood)
param["step_eta"]= 0.02;         # Learning rate for the M step
param["flag_imp_sampling"] = 0;  # Use importance sampling when sampling negative classes?
param["computePredTrain"] = 0;   # Compute predictions on training data? (it may be expensive)

mat"addpath ~/Competitors/augment-reduce/src"
mat_params = mat"[data param] = get_params_preprocess_data($param, $dataset_name, $data_path)";
# data_j = @mget data
param_j = @mget param
mat"data.X = sparse($X)";
mat"data.Y = $(Float64.(y))";
mat"data.test.X = sparse($X_test)";
mat"data.test.Y = $(Float64.(y_test))"
# param_j["B"] = minibatchsize; param_j["maxIter"] = nIterations;

mat"addpath ~/Competitors/augment-reduce/src/aux"
mat"addpath ~/Competitors/augment-reduce/src/infer"
# if !isdir(param["output_path"])
    # mkdir(param["output_path"])
# end

mat"rand('seed',0)"
mat"randn('seed',0)";

mat"metrics = run_classification(data, $param_j)"
met = @mget metrics
