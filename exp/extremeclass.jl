using OMGP
using DelimitedFiles#, CSV
using PyCall
using SpecialFunctions
using Distances, LinearAlgebra, Distributions,StatsBase
@pyimport sklearn.model_selection as sp
function initial_lengthscale(X)
    if size(X,1) > 10000
        D = pairwise(SqEuclidean(),X[sample(1:size(X,1),10000,replace=false),:]')
    else
        D = pairwise(SqEuclidean(),X')
    end

    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
function get_Dataset(datasetname::String)
    println("Getting dataset")
    data = readdlm("../data/"*datasetname*".csv",',')
    # data = Matrix{Float64}(CSV.read("../data/"*datasetname*".csv",header=false))
    X = data[:,1:end-1]; y = floor.(Int64,data[:,end]);
    println("Dataset loaded")
    return (X,y,datasetname)
end
function LogIt(model::OMGP.GPModel,iter)
    if in(iter,iter_points)
        a = Vector{Any}(undef,7)
        a[1] = time_ns()
        # y_p = OMGP.multiclasspredict(model,X_test,true)
        y_p = OMGP.multiclasspredictproba(model,X_test,false)
        # y_exp = OMGP.multiclasssoftmax(model,X_test,false)
        a[2] = TestAccuracy(model,y_test,y_p)
        loglike = LogLikelihood(model,y_test,y_p)
        # loglike_exp = LogLikelihood(model,y_test,y_exp)
        a[3] = mean(loglike)
        a[4] = median(loglike)
        a[5] = -OMGP.ELBO(model)

        a[7] = 0#[OMGP.KernelFunctions.getvalue(k.param) for k in model.kernel]
        println("Iteration $iter : Acc is $(a[2]), MeanL is $(a[3])")
        # println("Variances :", [OMGP.getvalue(k.variance) for k in model.kernel])
        a[6] = time_ns()
        push!(LogArrays,a)
    end
end
function TestAccuracy(model::OMGP.GPModel, y_test, y_predic)
    score = 0
    # println(y_predic,y_test)
    for i in 1:length(y_test)
      # println("$i : $(y_test[i]+1) => $(y_predic[i])")
        if (model.class_mapping[argmax(y_predic[i])]) == y_test[i]
            score += 1
        end
    end
    return score/length(y_test)
end
function LogLikelihood(model::OMGP.GPModel,y_test,y_predic)
    return [log(y_predic[i][model.ind_mapping[y_t]]) for (i,y_t) in enumerate(y_test)]
end
function TreatTime(init_time,before_comp,after_comp)
    before_comp = before_comp .- init_time; after_comp = after_comp .- init_time;
    diffs = after_comp-before_comp;
    for i in 2:length(before_comp)
        before_comp[i:end] .-= diffs[i-1]
    end
    return before_comp*1e-9
end


# (X,y,dataset) = get_Dataset("letter")
# X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.1)
(X,y,dataset) = get_Dataset("omniglot")
(X_test,y_test,dataset) = get_Dataset("omniglot_test")


l=initial_lengthscale(X)
println("Lengthscale estimated")
####PARAMETERS###

# kernel = RBFKernel([l],dim=size(X,2))
kernel = RBFKernel(l)
iter_points= vcat(3:10:99,100:100:999,1e3:1e3:(1e4-1),1e4:1e4:1e5)
M = 100; bsize = 100
doFull = true; doStoch = true; doAT = true;


# if doFull
# global LogArrays = Array{Any,1}()
# println("Starting training without class subsampling")
# global smodel = OMGP.SparseMultiClass(X,y,KStochastic=false,VerboseLevel=3,kernel=kernel,m=M,Autotuning=doAT,AutotuningFrequency=1,Stochastic=true,batchsize=bsize,IndependentGPs=false)
#
# global init_tfull = time_ns()
# smodel.train(iterations=50,callback=LogIt)
# y_spred = smodel.predict(X_test)[1]
# println("Sparse predictions computed")
# sparse_score=0
# for (i,pred) in enumerate(y_spred)
#     if pred == y_test[i]
#         global sparse_score += 1
#     end
# end
# println("Sparse model Accuracy is $(sparse_score/length(y_test))")#" in $t_sparse s")
# global LogFull = deepcopy(LogArrays)
# end


# if doStoch
global LogArrays = Array{Any,1}()
println("Starting training with class subsampling")
global ssmodel = OMGP.SparseMultiClass(X,y,KStochastic=true, nClassesUsed=20,VerboseLevel=3,kernel=kernel,m=M,Autotuning=doAT,AutotuningFrequency=1,Stochastic=true,batchsize=bsize,IndependentGPs=false)
global init_tstoch = time_ns()
ssmodel.train(iterations=100,callback=LogIt)

y_ssparse, = ssmodel.predict(X_test)
println("Sparse predictions computed")
ssparse_score=0
for (i,pred) in enumerate(y_ssparse)
    if pred == y_test[i]
        global ssparse_score += 1
    end
end
println("Super Sparse model Accuracy is $(ssparse_score/length(y_test))")
global LogStoch = deepcopy(LogArrays)
# end


LogFull = hcat(LogFull...)
Tfull = TreatTime(init_tfull,LogFull[1,:],LogFull[6,:])
Accfull = LogFull[2,:]
MeanLfull = LogFull[3,:]

LogStoch = hcat(LogStoch...)
Tstoch = TreatTime(init_tstoch,LogStoch[1,:],LogStoch[6,:])
Accstoch = LogStoch[2,:]
MeanLstoch = LogStoch[3,:]


# using Plots
# pyplot()
# # p1 = plot(Tfull,Accfull)
# p1 = plot(Tstoch,Accstoch)
# # p2 = plot(Tfull,MeanLfull)
# p2 = plot(Tstoch,MeanLstoch)
# plot(p1,p2)
