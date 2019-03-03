# Paper_Experiment_Functions.jl
#= ---------------- #
Set of functions for efficient testing.
# ---------------- =#



# module TestFunctions
  using DelimitedFiles, HDF5#, CSV
  using PyCall
  using Distances, LinearAlgebra, Distributions,StatsBase, GradDescent
  # using MATLAB
  using AugmentedGaussianProcesses
  @pyimport gpflow
  @pyimport tensorflow as tf
  # mat"addpath ~/Competitors/augment-reduce/src"
  # mat"addpath ~/Competitors/augment-reduce/src/aux"
  # mat"addpath ~/Competitors/augment-reduce/src/infer"
  # @pyimport TTGP.projectors as ttgpproj
  # @pyimport TTGP.covariance as ttgpcovariance
  # @pyimport TTGP.gpc_runner as ttgpcrun
  @pyimport sklearn.datasets as sk
  @pyimport sklearn.model_selection as sp
  using RCall
  R"rm(list = ls())"#Clear R REPL
  if doStochastic
    R"source('../src/sepMGPC_stochastic.R')"
    println("Loaded Stochastic R script")
  else
    R"source('../src/sepMGPC_batch.R')"
    println("Loaded Batch R script")
  end

  # export TestingModel
  # export get_Dataset
  # export DefaultParameters, SCPMCParameters, SVGPMCParameters, ARMCParameters, TTGPMCParameters
  # export CreateModel, TrainModel, RunTests, ProcessResults, PrintResults, WriteResults
  # export ComputePrediction, ComputePredictionAccuracy

function get_Dataset(datasetname::String)
    println("Getting dataset")
    data = h5read("../data/"*datasetname*".h5","data/data")
    # data = Matrix{Float64}(CSV.read("../data/"*datasetname*".csv",header=false))
    X = data[:,1:end-1]; y = floor.(Int64,data[:,end]);
    println("Dataset loaded")
    return (X,y,datasetname)
end

function get_train_test(datasetname::String)
    println("Getting dataset")
    X_train = h5read("../data/"*datasetname*".h5","data/X_train")
    X_test = h5read("../data/"*datasetname*".h5","data/X_test")
    y_train = Int64.(h5read("../data/"*datasetname*".h5","data/y_train"))
    y_test = Int64.(h5read("../data/"*datasetname*".h5","data/y_test"))
    println("Dataset loaded")
    return (X_train,y_train,X_test,y_test,datasetname)
end

function initial_lengthscale(X)
    if size(X,1) > 10000
        D = pairwise(SqEuclidean(),X[sample(1:size(X,1),10000,replace=false),:]')
    else
        D = pairwise(SqEuclidean(),X')
    end

    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
#Datatype for containing the model, its results and its parameters
mutable struct TestingModel
  MethodName::String #Name of the method
  DatasetName::String #Name of the dataset
  ExperimentType::String #Type of experiment
  MethodType::String #Type of method used ("SVM","BSVM","OMGP","SVGPC")
  Param::Dict{String,Any} #Some paramters to run the method
  Results::Dict{String,Any} #Saved results
  Model::Any
  TestingModel(methname,dataset,exp,methtype) = new(methname,dataset,exp,methtype,Dict{String,Any}(),Dict{String,Any}())
  TestingModel(methname,dataset,exp,methtype,params) = new(methname,dataset,exp,methtype,params,Dict{String,Any}())
end

include("initial_parameters.jl")


#Create a model given the parameters passed in p
function CreateModel!(tm::TestingModel,i,X,y) #tm testing_model, p parameters
    y_cmap = countmap(y)
    if tm.MethodType == "BCGPMC"
        tm.Model[i] = AugmentedGaussianProcesses.MultiClass(X,y;kernel=tm.Param["Kernel"],Autotuning=tm.Param["Autotuning"],AutotuningFrequency=tm.Param["ATFrequency"],ϵ=tm.Param["ϵ"],
            verbose=tm.Param["Verbose"],μ_init=tm.Param["FixedInitialization"] ? Float64.(zero(y)) : [0.0],IndependentGPs=tm.Param["independent"])
    elseif tm.MethodType == "SCGPMC" || tm.MethodType == "HSCGPMC"
        # tm.Param["time_init"] = @elapsed
        tm.Model[i] = AugmentedGaussianProcesses.SparseMultiClass(X,y;Stochastic=tm.Param["Stochastic"],batchsize=tm.Param["BatchSize"],m=tm.Param["M"],
            kernel=tm.Param["Kernel"],Autotuning=tm.Param["Autotuning"],OptimizeIndPoints=tm.Param["PointOptimization"],
            AutotuningFrequency=tm.Param["ATFrequency"],AdaptiveLearningRate=tm.Param["ALR"],κ_s=tm.Param["κ_s"],τ_s = tm.Param["τ_s"],ϵ=tm.Param["ϵ"],
            SmoothingWindow=tm.Param["Window"],verbose=tm.Param["Verbose"],μ_init=tm.Param["FixedInitialization"] ? zeros(tm.Param["M"]) : [0.0],IndependentGPs=tm.Param["independent"])
    elseif tm.MethodType == "SVGPMC"
        Z = AugmentedGaussianProcesses.KMeansInducingPoints(X,tm.Param["M"],nMarkov=10)
        # tm.Param["time_init"] = @elapsed A = Ind_KMeans.([y_cmap],[y],[X],tm.Param["M"],0:(tm.Param["nClasses"]-1))
        if tm.Param["Stochastic"]
            #Stochastic Sparse SVGPC model
            println("Creating stochastic SVGP")
            # tm.Param["time_init"] += @elapsed
            tm.Model[i] = gpflow.models[:SVGP](X, Float64.(reshape(y,(length(y),1))),kern=deepcopy(tm.Param["Kernel"]),likelihood=gpflow.likelihoods[:MultiClass](tm.Param["nClasses"]),num_latent=tm.Param["nClasses"],Z=Z,minibatch_size=tm.Param["BatchSize"])
        else
            #Sparse SVGPC model
            println("Creating full batch SVGP")
            # tm.Param["time_init"] += @elapsed
            tm.Model[i] = gpflow.models[:SVGP](X, Float64.(reshape(y,(length(y),1))),kern=deepcopy(tm.Param["Kernel"]),likelihood=gpflow.likelihoods[:MultiClass](tm.Param["nClasses"]),num_latent=tm.Param["nClasses"],Z=Z)
        end
    elseif tm.MethodType == "TTGPMC"
        tm.Model[i] = 0
    elseif tm.MethodType == "EPGPMC"
        tm.Model[i] = 0
    elseif tm.MethodType == "ARMC"
        tm.Model[i] = Dict{String,Any}(); tm.Model[i]["method_name"] = "softmax_a&r";
        tm.Model[i]["maxIter"]=tm.Param["maxIter"]; tm.Model[i]["B"] = tm.Param["BatchSize"]
        tm.Model[i]["flag_mexFile"]=1; tm.Model[i]["s2prior"] = Inf; tm.Model[i]["step_eta"] = 0.02;
        tm.Model[i]["flag_imp_sampling"] = 0; tm.Model[i]["computePredTrain"] = 0;
    end
end


function run_nat_grads_with_adam(model,iterations; ind_points_fixed=true, kernel_fixed =false, callback=nothing , Stochastic = true)
    # we'll make use of this later when we use a XiTransform

    gamma_start = 1e-5;
    if Stochastic
        gamma_max = 1e-1;    gamma_step = 10^(0.1); gamma_fallback = 1e-2;
    else
        gamma_max = 1e-1;    gamma_step = 10^(0.1); gamma_fallback = 1e-2;
    end
    gamma = tf.Variable(gamma_start,dtype=tf.float64);    gamma_incremented = tf.where(tf.less(gamma,gamma_max),gamma*gamma_step,gamma_max)
    op_increment_gamma = tf.assign(gamma,gamma_incremented)
    op_gamma_fallback = tf.assign(gamma,gamma*gamma_fallback);
    sess = model[:enquire_session]();    sess[:run](tf.variables_initializer([gamma]));
    var_list = [(model[:q_mu], model[:q_sqrt])]
    # we don't want adam optimizing these
    model[:q_mu][:set_trainable](false)
    model[:q_sqrt][:set_trainable](false)
    #
    ind_points_fixed ? model[:feature][:set_trainable](false) : nothing
    kernel_fixed ? model[:kern][:set_trainable](false) : nothing
    op_natgrad = gpflow.training[:NatGradOptimizer](gamma=gamma)[:make_optimize_tensor](model, var_list=var_list)
    op_adam=0

    if !(ind_points_fixed && kernel_fixed)
        op_adam = gpflow.train[:AdamOptimizer]()[:make_optimize_tensor](model)
    end

    for i in 1:(10*iterations)
        try
            sess[:run](op_natgrad);sess[:run](op_increment_gamma)
        catch e
          if isa(e,InterruptException)
                    println("Training interrupted by user at iteration $i");
                    break;
          else
            g = sess[:run](gamma)
            # println("Gamma $g on iteration $i is too big: Falling back to $(g*gamma_fallback)")
            sess[:run](op_gamma_fallback)
          end
        end
        if op_adam!=0
            sess[:run](op_adam)
        end
        if i % 100 == 0
            println("$i gamma=$(sess[:run](gamma)) ELBO=$(sess[:run](model[:likelihood_tensor]))")
        end
        if callback!= nothing
            callback(model,sess,i)
        end
    end
    model[:anchor](sess)
end

function trainhybrid(model,iterationsaug,iterations,callback)
    model.train(iterations=iterationsaug,callback=callback)
    new_model = AugmentedGaussianProcesses.SparseLogisticSoftMaxMultiClass(model.X,model.y,Stochastic=model.Stochastic,m=model.nFeatures,batchsize=model.nSamplesUsed,kernel=model.kernel[1],IndependentGPs=model.IndependentGPs,verbose=3,nEpochs=50,optimizer=0.5,Autotuning=true)
    for field in fieldnames(typeof(model))
        if field != :prev_params && field != :K_map && field != :g && field!= :h && field!= :τ && field != :Knn && field != :invK   && field != :train && field != :fstar && field != :predict && field != :predictproba && field != :elbo && field != :Autotuning
            @eval $new_model.$field = $model.$field
        end
    end
    println(model.kernel)
    model = new_model
    model.train(iterations=iterations,callback=callback)
    return model
end

"Function to obtain the weighted KMeans for one class"
function Ind_KMeans(N_inst,Y,X,m,i)
    nSamples = size(X,1)
    K_corr = nSamples/N_inst[i]-1.0
    weights = ones(nSamples)
    weights[Y.==i] = weights[Y.==i].*(K_corr-1.0).+(1.0)
    return AugmentedGaussianProcesses.KMeansInducingPoints(X,m,nMarkov=10,kweights=weights)
end

function TrainModel!(tm::TestingModel,i,X,y,X_test,y_test,iterations,iter_points)
    LogArrays = Array{Any,1}()
    if tm.MethodType == "SCGPMC"
        function LogIt(model,iter)
            if in(iter,iter_points)
                a = Vector{Any}(undef,9)
                a[1] = time_ns()
                AugmentedGaussianProcesses.computeMatrices!(model)
                # y_p = OMGP.multiclasspredict(model,X_test,true)
                # y_p = multiclasspredictproba(model,X_test,false)
                y_p = proba_y(model,X_test)
                # y_exp = OMGP.multiclasssoftmax(model,X_test,false)
                a[2] = TestAccuracy(model,y_test,y_p)
                loglike = LogLikelihood(model,y_test,y_p)
                # loglike_exp = LogLikelihood(model,y_test,y_exp)
                a[3] = mean(loglike)
                a[4] = median(loglike)
                a[5] = ELBO(model)
                y_pred = Float64.(zero(y_test))
                for i in unique(y_test)
                    y_pred[y_test.==i] = y_p[Symbol(i)][y_test.==i]
                end
                a[7] = rcopy(R"multiclass.roc($y_test,$y_pred)$auc")
                println("Iteration $iter : Acc is $(a[2]), MeanL is $(a[3])")
                # a[8],a[9] = calibration_R(y_test,y_p)
                a[8],a[9] = calibration(y_test,y_p)
                a[8] = mean(a[8])
                a[9] = mean(a[9])
                a[6] = time_ns()
                push!(LogArrays,a)
            end
        end
        if tm.MethodType == "HSCGPMC"
            trainhybrid(tm.Model[i],tm.Param["nConjugateSteps"],iterations,LogIt)
        else
            train!(tm.Model[i],iterations=iterations,callback=LogIt)
        end
    elseif tm.MethodType == "SVGPMC"
      function pythonlogger(model,session,iter)
            if in(iter,iter_points)
                a = Vector{Any}(undef,9)
                a[1] = time_ns()
                y_p = model[:predict_y](X_test)[1]
                loglike = LogLikelihood(y_test,y_p)
                a[2] = TestAccuracy(y_test,y_p)
                a[3] = mean(loglike)
                a[4] = median(loglike)
                a[5] = session[:run](model[:likelihood_tensor])
                y_pred = Float64.(zero(y_test))
                bias = count(unique(y_test).==0)!=0
                for i in unique(y_test)
                    y_pred[y_test.==i] = y_p[y_test.==i,i+bias]
                end
                a[7] = rcopy(R"multiclass.roc($y_test,$y_pred)$auc")
                println("Iteration $iter : Acc is $(a[2]), MeanL is $(a[3])")
                # a[8],a[9] = calibration_R(y_test,y_p,gpflow=true)
                a[8],a[9] = calibration(y_test,y_p,gpflow=true)
                a[8] = mean(a[8])
                a[9] = mean(a[9])
                a[6] = time_ns()
                push!(LogArrays,a)
            end
      end
      run_nat_grads_with_adam(tm.Model[i], iterations; ind_points_fixed=!tm.Param["PointOptimization"], kernel_fixed =!tm.Param["Autotuning"],callback=pythonlogger,Stochastic=tm.Param["Stochastic"])
    elseif tm.MethodType == "EPGPMC"
        y_cmap = countmap(y)
        m = tm.Param["M"]; bs = tm.Param["BatchSize"]; pointopt=!tm.Param["PointOptimization"]; autotu=tm.Param["Autotuning"];
        #tm.Param["time_init"] = @elapsed
        Xbar_ini = Ind_KMeans.([y_cmap],[y],[X],m,0:(tm.Param["nClasses"]-1))
        if tm.Param["Stochastic"]
          tm.Model[i] = R"epMGPCInternal($X, $(y
        .+1), m = $m, X_test = $X_test, Y_test= $(y_test.+1), n_minibatch = $bs, max_iters=$iterations, indpoints= $pointopt, autotuning=$autotu, Xbar_ini=$Xbar_ini)"
        else
          tm.Model[i] = R"epMGPCInternal($X, $(y
        .+1), m = $m, X_test = $X_test, Y_test= $(y_test.+1), max_iters=$iterations, indpoints= $pointopt, autotuning=$autotu)"
        end
        LogArrays = Matrix(rcopy(tm.Model[i][:log_table]))[:,2:end]
        # LogArrays[:,1] = LogArrays[:,1] .+ tm.Param["time_init"]
    elseif tm.MethodType == "TTGPMC"
      stable = false
      while !stable
        try
          @pywith tf.Graph()[:as_default]() begin
            proj = ttgpproj.Identity(D=tm.Param["nFeatures"])
            kernel = ttgpcovariance.SE_multidim(tm.Param["nClasses"],0.9,tm.Param["theta"],tm.Param["γ"],proj)

            runner = ttgpcrun.GPCRunner(tm.Param["N_grid"],tm.Param["mu_TT"],kernel,X=X,y=(y.+1)./2,X_test=X_test,y_test=(y_test.+1)./2,
                                       lr=tm.Param["lr"],batch_size=tm.Param["BatchSize"],batch_test=false,n_epoch=iterations)
            all_pred,LogArrays = runner[:run_experiment]()
          end
          stable = true
        catch e
          # println(e)
          println("Number of inducing points is not working, adding 1")
          tm.Param["N_grid"] += 1
        end
    end
    elseif tm.MethodType == "ARMC"
        t0 = time_ns()
        # mat"data.X = sparse($X)";mat"data.Y = $(Float64.(y))";
        # mat"data.test.X = sparse($X_test)"; mat"data.test.Y = $(Float64.(y_test))"
        # mat"metrics = run_classification(data, $param_j)"
        # LogArrays = @mget metrics
    end
    return LogArrays
end

function TreatTime(init_time,before_comp,after_comp)
    before_comp = before_comp .- init_time; after_comp = after_comp .- init_time;
    diffs = after_comp-before_comp;
    for i in 2:length(before_comp)
        before_comp[i:end] .-= diffs[i-1]
    end
    return before_comp*1e-9
end

#Run tests accordingly to the arguments and s

function ProcessResults(tm::TestingModel,iFold)
    #Find maximum length
    iFold = length(tm.Results["Time"])
    NMax = maximum(length.(tm.Results["Time"]))
    NFolds = length(tm.Results["Time"])
    Mtime = zeros(NMax); time= []
    Macc = zeros(NMax); acc= []
    Mmeanl = zeros(NMax); meanl= []
    Mmedianl = zeros(NMax); medianl= []
    Melbo = zeros(NMax); elbo = []
    Mauc = zeros(NMax); auc = []
    Mece = zeros(NMax); ece = []
    Mmce = zeros(NMax); mce = []
    for i in 1:iFold
        DiffN = NMax - length(tm.Results["Time"][i])
        if DiffN != 0
            time = [tm.Results["Time"][i];tm.Results["Time"][i][end]*ones(DiffN)]
            acc = [tm.Results["Accuracy"][i];tm.Results["Accuracy"][i][end]*ones(DiffN)]
            meanl = [tm.Results["MeanL"][i];tm.Results["MeanL"][i][end]*ones(DiffN)]
            medianl = [tm.Results["MedianL"][i];tm.Results["MedianL"][i][end]*ones(DiffN)]
            elbo = [tm.Results["ELBO"][i];tm.Results["ELBO"][i][end]*ones(DiffN)]
            auc = [tm.Results["AUC"][i];tm.Results["AUC"][i][end]*ones(DiffN)]
            ece = [tm.Results["ECE"][i];tm.Results["ECE"][i][end]*ones(DiffN)]
            mce = [tm.Results["MCE"][i];tm.Results["MCE"][i][end]*ones(DiffN)]
        else
            time = tm.Results["Time"][i];
            acc = tm.Results["Accuracy"][i];
            meanl = tm.Results["MeanL"][i];
            medianl = tm.Results["MedianL"][i];
            elbo = tm.Results["ELBO"][i];
            auc = tm.Results["AUC"][i];
            ece = tm.Results["ECE"][i];
            mce = tm.Results["MCE"][i];
        end
        Mtime = hcat(Mtime,time)
        Macc = hcat(Macc,acc)
        Mmeanl = hcat(Mmeanl,meanl)
        Mmedianl = hcat(Mmedianl,medianl)
        Melbo = hcat(Melbo,elbo)
        Mauc = hcat(Mauc,auc)
        Mece = hcat(Mece,ece)
        Mmce = hcat(Mmce,mce)
    end
    if size(Mtime,2)!=2
      Mtime[:,2] = Mtime[:,3]
    end
    Mtime = Mtime[:,2:end];  Macc = Macc[:,2:end]
    Mmeanl = Mmeanl[:,2:end]; Mmedianl = Mmedianl[:,2:end]
    Melbo = Melbo[:,2:end]; Mauc = Mauc[:,2:end];
    tm.Results["Time"] = Mtime;
    tm.Results["Accuracy"] = Macc;
    tm.Results["MeanL"] = Mmeanl
    tm.Results["MedianL"] = Mmedianl
    tm.Results["ELBO"] = Melbo
    tm.Results["AUC"] = Mauc
    tm.Results["ECE"] = Mece
    tm.Results["MCE"] = Mmce
    tm.Results["Processed"]= [vec(mean(Mtime,dims=2)) vec(std(Mtime,dims=2)) vec(mean(Macc,dims=2)) vec(std(Macc,dims=2)) vec(mean(Mmeanl,dims=2)) vec(std(Mmeanl,dims=2)) vec(mean(Mmedianl,dims=2)) vec(std(Mmedianl,dims=2)) vec(mean(Melbo,dims=2)) vec(std(Melbo,dims=2)) vec(mean(Mauc,dims=2)) vec(std(Mauc,dims=2)) vec(mean(Mece,dims=2)) vec(std(Mece,dims=2)) vec(mean(Mmce,dims=2)) vec(std(Mmce,dims=2))]
end

function PrintResults(results,method_name,writing_order)
  println("Model $(method_name) : ")
  i = 1
  for category in writing_order
    println("$category : $(results[i*2-1]) ± $(results[i*2])")
    i+=1
  end
end


function WriteLastProba(tm::TestingModel,location,X_test,y_test)
    fold = String(location*"/"*(doAutotuning ? "AT_" : "")*(doStochastic ? "S_" : "")*"Experiment")
    if !isdir(fold); mkdir(fold); end;
    fold = fold*"/"*tm.DatasetName*"Dataset"
    y_p = []
    if tm.MethodType == "SCGPMC"
        y_p = proba_y(tm.Model[1],X_test)
        reorder = sortperm(parse.(Int64,string.(names(y_p))))
        y_p = Matrix(y_p)[:,reorder]
    elseif tm.MethodType == "EPGPMC"
        y_p = R"predictMGPC($(tm.Model[1]),$(X_test))$prob"
        y_p = Matrix(rcopy(y_p))
    elseif tm.MethodType == "SVGPMC"
        y_p = tm.Model[1][:predict_y](X_test)[1]
    end
    y_p = hcat(y_test,y_p)
    if !isdir(fold); mkdir(fold); end;
    writedlm(String(fold*"/y_prob_"*tm.MethodName*(tm.MethodType=="SCGPMC" ? (!tm.Param["independent"] ? "_shared" : "") : "")*".txt"),y_p)
end

function WriteResults(tm::TestingModel,location,writing_order,kfold)
    fold = String(location*"/"*(doAutotuning ? "AT_" : "")*(doStochastic ? "S_" : "")*(kfold ? "K_" : "")*"Experiment")
    if !isdir(fold); mkdir(fold); end;
    fold = fold*"/"*tm.DatasetName*"Dataset"
    # labels=Array{String,1}(undef,length(writing_order)*2)
    # labels[1:2:end-1,:] = writing_order.*"_mean"
    # labels[2:2:end,:] =  writing_order.*"_std"
    if !isdir(fold); mkdir(fold); end;
    writedlm(String(fold*"/Results_"*tm.MethodName*(tm.MethodType=="SCGPMC" ? (!tm.Param["independent"] ? "_shared" : "") : "")*".txt"),tm.Results["Processed"])
end

#Return Accuracy on test set
function TestAccuracy(model, y_test, y_predic)
    score = 0
    # println(y_predic,y_test)
    for i in 1:length(y_test)
      # println("$i : $(y_test[i]+1) => $(y_predic[i])")
        if parse(Int64,string(argmax(y_predic[i,:]))) == y_test[i]
            score += 1
        end
    end
    return score/length(y_test)
end

function TestAccuracy(y_test, y_predic)
    score = 0
    # println(y_predic,y_test)
    for i in 1:length(y_test)
        if (argmax(y_predic[i,:])-1) == y_test[i]
            score += 1
        end
    end
    return score/length(y_test)
end


function LogLikelihood(model,y_test,y_predic)
    return [log(y_predic[Symbol(y_t)][i]) for (i,y_t) in enumerate(y_test)]
end

function LogLikelihood(y_test,y_predic)
    return [log(y_predic[i,y_t+1]) for (i,y_t) in enumerate(y_test)]
end


function WriteLastStateParameters(testmodel,top_fold,X_test,y_test,i)
    if isa(testmodel.Model[i],AugmentedGaussianProcesses.GPModel)
        if !isdir(top_fold); mkdir(top_fold); end;
        top_fold = top_fold*"/"*testmodel.DatasetName*"_SavedParams"
        if !isdir(top_fold); mkdir(top_fold); end;
        top_fold = top_fold*"/"*testmodel.MethodName
        if !isdir(top_fold); mkdir(top_fold); end;
        writedlm(top_fold*"/mu"*"_$i",testmodel.Model[i].μ)
        writedlm(top_fold*"/sigma"*"_$i",testmodel.Model[i].ζ)
        writedlm(top_fold*"/c"*"_$i",testmodel.Model[i].α)
        writedlm(top_fold*"/X_test"*"_$i",X_test)
        writedlm(top_fold*"/y_test"*"_$i",y_test)
        if isa(testmodel.Model[i],AugmentedGaussianProcesses.SparseModel)
            writedlm(top_fold*"/ind_points"*"_$i",testmodel.Model[i].inducingPoints)
        end
        if isa(testmodel.Model[i],AugmentedGaussianProcesses.NonLinearModel)
            writedlm(top_fold*"/kernel_param"*"_$i",broadcast(getfield,testmodel.Model[i].kernel,:param))
            writedlm(top_fold*"/kernel_coeff"*"_$i",broadcast(getfield,testmodel.Model[i].kernel,:coeff))
            writedlm(top_fold*"/kernel_name"*"_$i",broadcast(getfield,testmodel.Model[i].kernel,:name))
        end
	println("Last state saved in $top_fold")
    end
end



function PlotResults(TestModels)
    nModels=length(TestModels)
    if nModels == 0; return; end;
    colors=["blue", "red", "green"]
    ps = []
    ##Accuracy
        p = plot()
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot!(p,results[1:step:end,1],1.0.-results[1:step:end,3],color=colors[iter],label=name,xaxis=:log,xlabel="Time [s]",ylabel="Test Error")
            # fill_between(results[1:step:end,1],results[1:step:end,3]-results[1:step:end,4]/sqrt(10),results[1:step:end,3]+results[1:step:end,4]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
        push!(ps,p)
    ##MeanL
        p = plot()
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot!(p,results[1:step:end,1],-results[1:step:end,5],color=colors[iter],label=name,xaxis=:log,xlabel="Time [s]",ylabel="Neg MeanL")
            # fill_between(results[1:step:end,1],results[1:step:end,5]-results[1:step:end,6]/sqrt(10),results[1:step:end,5]+results[1:step:end,6]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
        push!(ps,p)
    ##MedianL
        p =plot()
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot!(p,results[1:step:end,1],-results[1:step:end,7],color=colors[iter],label=name,xaxis=:log,xlabel="Time [s]",ylabel="Neg MedianL")
            # fill_between(results[1:step:end,1],results[1:step:end,7]-results[1:step:end,8]/sqrt(10),results[1:step:end,7]+results[1:step:end,8]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
        push!(ps,p)
    ##AUC
        p = plot()
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot!(p,results[1:step:end,1],results[1:step:end,11],color=colors[iter],label=name,xaxis=:log,xlabel="Time [s]",ylabel="AUC")
            # fill_between(results[1:step:end,1],results[1:step:end,11]-results[1:step:end,12]/sqrt(10),results[1:step:end,11]+results[1:step:end,12]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
        push!(ps,p)
    ##ECE
        p=plot()
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot!(p,results[1:step:end,1],results[1:step:end,13],color=colors[iter],label=name,xaxis=:log,yaxis=:log,xlabel="Time [s]",ylabel="ECE")
            # fill_between(results[1:step:end,1],results[1:step:end,13]-results[1:step:end,14]/sqrt(10),results[1:step:end,13]+results[1:step:end,14]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
        push!(ps,p)
    ##MCE
        p = plot()
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot!(p,results[1:step:end,1],results[1:step:end,15],color=colors[iter],label=name,xaxis=:log,yaxis=:log,xlabel="Time [s]",ylabel="MCE")
            # fill_between(results[1:step:end,1],results[1:step:end,15]-results[1:step:end,16]/sqrt(10),results[1:step:end,15]+results[1:step:end,16]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
        push!(ps,p)
    display(title!(plot(ps...),"Convergence Results"))
end


# end #end of module
