# Paper_Experiment_Functions.jl
#= ---------------- #
Set of functions for efficient testing.
# ---------------- =#
using DelimitedFiles



module TestFunctions
  using PyCall
  using Distributions
  using MATLAB
  import OMGP
  @pyimport gpflow
  @pyimport tensorflow as tf
  mat"addpath ~/Competitors/augment-reduce/src"
  mat"addpath ~/Competitors/augment-reduce/src/aux"
  mat"addpath ~/Competitors/augment-reduce/src/infer"
  @pyimport TTGP.projectors as ttgpproj
  @pyimport TTGP.covariance as ttgpcovariance
  @pyimport TTGP.gpc_runner as ttgpcrun
  @pyimport sklearn.datasets as sk
  @pyimport sklearn.model_selection as sp
  using RCall
  R"source('EP_stochastic.R')"


function get_Dataset(datasetname::String)
    data = readdlm("../data/"*datasetname)
    X = data[:,1:end-1]; y = floor.(Int64,data[:,end]);
    return (X,y,datasetname)
end
export TestingModel
export DefaultParameters, XGPMCParameters, SVGPMCParameters, ARMCParameters, TTGPMCParameters
export CreateModel, TrainModel, RunTests, ProcessResults, PrintResults, WriteResults
export ComputePrediction, ComputePredictionAccuracy

#Datatype for containing the model, its results and its parameters
type TestingModel
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

#Create a default dictionary
function DefaultParameters()
  param = Dict{String,Any}()
  param["ϵ"]= 1e-8 #Convergence criteria
  param["BatchSize"] = 10 #Number of points used for stochasticity
  param["Kernel"] = "rbf" # Kernel function
  param["Θ"] = 1.0 # Hyperparameter for the kernel function
  param["γ"] = 1.0 #Variance of introduced noise
  param["M"] = 32 #Number of inducing points
  param["Window"] = 5 #Number of points used to check convergence (smoothing effect)
  param["Verbose"] = 0 #Verbose
  param["Autotuning"] = false
  param["ConvCriter"] = "HOML"
  param["PointOptimization"] = false
  param["FixedInitialization"] = true
  return param
end

#Create a default parameters dictionary for XGPC
function XGPMCParameters(;Stochastic=true,Sparse=true,ALR=true,Autotuning=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["Stochastic"] = Stochastic #Is the method stochastic
  param["Sparse"] = Sparse #Is the method using inducing points
  param["ALR"] = ALR #Is the method using adpative learning rate (in case of the stochastic case)
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ATFrequency"] = 1 #Number of iterations between every autotuning
  param["κ_s"] = 1.0;  param["τ_s"] = 100; #Parameters for learning rate of Stochastic gradient descent when ALR is not used
  param["ϵ"] = main_param["ϵ"]; param["Window"] = main_param["Window"]; #Convergence criteria (checking parameters norm variation on a window)
  param["ConvCriter"] = main_param["ConvCriter"]
  param["Kernels"] = OMGP.RBFKernel(main_param["Θ"]) #Kernel creation (standardized for now)
  param["Verbose"] = if typeof(main_param["Verbose"]) == Bool; main_param["Verbose"] ? 2 : 0; else; param["Verbose"] = main_param["Verbose"]; end; #Verbose
  param["BatchSize"] = main_param["BatchSize"] #Number of points used for stochasticity
  param["FixedInitialization"] = main_param["FixedInitialization"]
  param["M"] = main_param["M"] #Number of inducing points
  param["γ"] = main_param["γ"] #Variance of introduced noise
  return param
end

#Create a default parameters dictionary for SVGPMC (similar to XGPMC)
function SVGPMCParameters(;Stochastic=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["Sparse"] = true
  if Sparse
    param["Stochastic"] = Stochastic
  else
    param["Stochastic"] = false
  end
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ϵ"] = main_param["ϵ"]
  param["Kernel"] = gpflow.kernels[:Sum]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=false),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  param["SmoothingWindow"] = main_param["Window"]
  param["ConvCriter"] = main_param["ConvCriter"]
  return param
end

#Create a default parameters dictionary for EPGPC (similar to XGPMC)
function EPGPMCParameters(;main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ϵ"] = main_param["ϵ"]
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  param["SmoothingWindow"] = main_param["Window"]
  param["ConvCriter"] = main_param["ConvCriter"]
  return param
end

#Create a default parameters dictionary for TTGPC (similar to XGPMC)
function TTGPMCParameters(;Stochastic=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ϵ"] = main_param["ϵ"]
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  param["SmoothingWindow"] = main_param["Window"]
  param["ConvCriter"] = main_param["ConvCriter"]
  return param
end


#Create a default parameters dictionary for ARMC (similar to XGPMC)
function ARMCParameters(;Stochastic=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["maxIter"]=main_param["maxIter"]
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  param["SmoothingWindow"] = main_param["Window"]
  param["ConvCriter"] = main_param["ConvCriter"]
  return param
end


#Create a model given the parameters passed in p
function CreateModel!(tm::TestingModel,i,X,y) #tm testing_model, p parameters
    if tm.MethodType == "BXGPMC"
        tm.Model[i] = OMGP.MultiClass(X,y;kernel=tm.Param["Kernels"],Autotuning=tm.Param["Autotuning"],AutotuningFrequency=tm.Param["ATFrequency"],ϵ=tm.Param["ϵ"],noise=tm.Param["γ"],
            VerboseLevel=tm.Param["Verbose"],μ_init=tm.Param["FixedInitialization"] ? zeros(y) : [0.0])
    elseif tm.MethodType == "SXGPMC"
        tm.Model[i] = OMGP.SparseMultiClass(X,y;Stochastic=tm.Param["Stochastic"],BatchSize=tm.Param["BatchSize"],m=tm.Param["M"],
            kernel=tm.Param["Kernels"],Autotuning=tm.Param["Autotuning"],OptimizeIndPoints=tm.Param["PointOptimization"],
            AutotuningFrequency=tm.Param["ATFrequency"],AdaptiveLearningRate=tm.Param["ALR"],κ_s=tm.Param["κ_s"],τ_s = tm.Param["τ_s"],ϵ=tm.Param["ϵ"],noise=tm.Param["γ"],
            SmoothingWindow=tm.Param["Window"],VerboseLevel=tm.Param["Verbose"],μ_init=tm.Param["FixedInitialization"] ? zeros(tm.Param["M"]) : [0.0])
    elseif tm.MethodType == "SVGPMC"
        if tm.Param["Stochastic"]
            #Stochastic Sparse SVGPC model
            tm.Model[i] = gpflow.models[:SVGP](X, y,kern=deepcopy(tm.Param["Kernel"]), likelihood=gpflow.likelihoods[:MultiClass](tm.Param["nClasses"]),num_latent=tm.Param["nClasses"],
             Z=OMGP.KMeansInducingPoints(X,tm.Param["M"],10), minibatch_size=tm.Param["BatchSize"])
        else
            #Sparse SVGPC model
            tm.Model[i] = gpflow.models[:SVGP](X, reshape((y+1)./2,(size(y,1),1)),kern=deepcopy(tm.Param["Kernel"]),likelihood=gpflow.likelihoods[:MultiClass](tm.Param["nClasses"]),num_latent=tm.Param["nClasses"],
             Z=OMGP.KMeansInducingPoints(X,tm.Param["M"],10))
        end
    elseif tm.MethodType == "TTGPMC"
        tm.Model[i] = 0
    elseif tm.MethodType == "EPGPC"
        tm.Model[i] = 0
    elseif tm.MethodType == "ARMC"
        tm.Model[i] = Dict{String,Any}(); tm.Model[i]["method_name"] = "softmax_a&r";
        tm.Model[i]["maxIter"]=tm.Param["maxIter"]; tm.Model[i]["B"] = tm.Param["BatchSize"]
        tm.Model[i]["flag_mexFile"]=1; tm.Model[i]["s2prior"] = Inf; tm.Model[i]["step_eta"] = 0.02;
        tm.Model[i]["flag_imp_sampling"] = 0; tm.Model[i]["computePredTrain"] = 0;
    end
end

function run_with_adam(model,iterations; ind_points_fixed=true, kernel_fixed =false, callback=nothing)
    # we'll make use of this later when we use a XiTransform
    sess = model[:enquire_session]()
    ind_points_fixed ? model[:feature][:set_trainable](false) : nothing
    kernel_fixed ? model[:kern][:set_trainable](false) : nothing
    op_adam = gpflow.train[:AdamOptimizer]()[:make_optimize_tensor](model)
    for iter in 1:iterations
        sess[:run](op_adam)
        callback(model,sess,iter)
        if iter % 100 == 0
            print("$iter gamma=$(sess[:run](gamma)) ELBO=$(sess[:run](model[:likelihood_tensor]))")
        end
    end
    model[:anchor](model[:enquire_session]())
end

function TrainModel!(tm::TestingModel,i,X,y,X_test,y_test,iterations,iter_points)
    LogArrays = Array{Any,1}()
    if typeof(tm.Model[i]) <: OMGP.GPModel
        function LogIt(model::OMGP.GPModel,iter)
            if in(iter,iter_points)
                a = zeros(6)
                a[1] = time_ns()
                y_sparse,loglike = model.predict(X_test)
                loglike = log.(loglike)
                a[2] = TestAccuracy(y_test,y_p)
                a[3] = TestMeanHoldOutLikelihood(loglike)
                a[4] = TestMedianHoldOutLikelihood(loglike)
                a[5] = OMGP.ELBO(model)
                println("Iteration $iter : Acc is $(a[2]), MeanL is $(a[4])")
                a[6] = time_ns()
                push!(LogArrays,a)
            end
        end
      tm.Model[i].train(iterations=iterations,callback=LogIt)
    elseif tm.MethodType == "SVGPMC"
      function pythonlogger(model,session,iter)
            if in(iter,iter_points)
                a = zeros(8)
                a[1] = time_ns()
                y_p = model[:predict_y](X_test)[1]
                a[2] = TestAccuracy(y_test,y_p)
                a[5] = session[:run](model[:likelihood_tensor])
                a[6] = time_ns()
                push!(LogArrays,a)
            end
      end
      run_adam_and_callback(tm.Model[i], iterations; ind_points_fixed=!tm.Param["PointOptimization"], kernel_fixed =!tm.Param["Autotuning"],callback=pythonlogger)
    elseif tm.MethodType == "EPGPC"
        m = tm.Param["M"]; bs = tm.Param["BatchSize"]; pointopt=tm.Param["PointOptimization"]; autotu=tm.Param["Autotuning"];
        ind_points = OMGP.KMeansInducingPoints(X,tm.Param["M"],10)
        tm.Model[i] = R"epGPCInternal($X, $y, inducingpoints=$ind_points, m = $m, X_test = $X_test, Y_test= $y_test, n_minibatch = $bs, max_iters=$iterations,indpointsopt= $pointopt, hyperparamopt=$autotu,callback=save_log)"
        LogArrays = rcopy(tm.Model[i][:log_table]).columns[2:end]
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
    elseif tm.MethodType == "ARMC"
        t0 = time_ns()
        mat"data.X = sparse($X)";mat"data.Y = $(Float64.(y))";
        mat"data.test.X = sparse($X_test)"; mat"data.test.Y = $(Float64.(y_test))"
        mat"metrics = run_classification(data, $param_j)"
        LogArrays = @mget metrics
    end
    return LogArrays
end

function TreatTime(init_time,before_comp,after_comp)
    before_comp = before_comp - init_time; after_comp = after_comp - init_time;
    diffs = after_comp-before_comp;
    for i in 2:length(before_comp)
        before_comp[i:end] -= diffs[i-1]
    end
    return before_comp*1e-9
end

#Run tests accordingly to the arguments and save them
function RunTests(tm::TestingModel,i,X,X_test,y_test;accuracy::Bool=true,brierscore::Bool=true,logscore::Bool=true,AUCscore::Bool=true,likelihoodscore::Bool=true,npoints::Integer=500)
  if accuracy
    tm.Results["Accuracy"][i]=TestAccuracy(y_test,ComputePrediction(tm,i,X,X_test))
  end
  y_predic_acc = 0
  if brierscore
    y_predic_acc = ComputePredictionAccuracy(tm,i, X, X_test)
    tm.Results["Brierscore"][i] = TestBrierScore(y_test,y_predic_acc)
  end
  if logscore
    if y_predic_acc == 0
      y_predic_acc = ComputePredictionAccuracy(tm,i, X, X_test)
    end
    tm.Results["-Logscore"][i]=TestLogScore(y_test,y_predic_acc)
  end
  if AUCscore
    if y_predic_acc == 0
        y_predic_acc = ComputePredictionAccuracy(tm,i,X,X_test)
    end
    tm.Results["AUCscore"][i] = TestAUCScore(ROC(y_test,y_predic_acc,npoints))
  end
  if likelihoodscore
      if y_predic_acc == 0
          y_predic_acc = ComputePredictionAccuracy(tm,i,X,X_test)
      end
      tm.Results["-MedianL"][i] = -TestMedianHoldOutLikelihood(HoldOutLikelihood(y_test,y_predic_acc))
      tm.Results["-MeanL"][i] = -TestMeanHoldOutLikelihood(HoldOutLikelihood(y_test,y_predic_acc))
  end
end


#Compute the mean and the standard deviation and assemble in one result
function ProcessResults(tm::TestingModel,writing_order)
  all_results = Array{Float64,1}()
  names = Array{String,1}()
  for name in writing_order
    result = [mean(tm.Results[name]), std(tm.Results[name])]
    all_results = vcat(all_results,result)
    names = vcat(names,name)
  end
  if haskey(tm.Results,"allresults")
    tm.Results["allresults"] = vcat(tm.Results["allresults"],all_results')
  else
    tm.Results["allresults"] = all_results'
  end
  if !haskey(tm.Results,"names")
    tm.Results["names"] = names
  end
end

function ProcessResultsConvergence(tm::TestingModel,iFold)
    #Find maximum length
    NMax = maximum(length.(tm.Results["Time"][1:iFold]))
    NFolds = length(tm.Results["Time"][1:iFold])
    Mtime = zeros(NMax); time= []
    Macc = zeros(NMax); acc= []
    Mmeanl = zeros(NMax); meanl= []
    Mmedianl = zeros(NMax); medianl= []
    Melbo = zeros(NMax); elbo = []
    for i in 1:iFold
        DiffN = NMax - length(tm.Results["Time"][i])
        if DiffN != 0
            time = [tm.Results["Time"][i];tm.Results["Time"][i][end]*ones(DiffN)]
            acc = [tm.Results["Accuracy"][i];tm.Results["Accuracy"][i][end]*ones(DiffN)]
            meanl = [tm.Results["MeanL"][i];tm.Results["MeanL"][i][end]*ones(DiffN)]
            medianl = [tm.Results["MedianL"][i];tm.Results["MedianL"][i][end]*ones(DiffN)]
            elbo = [tm.Results["ELBO"][i];tm.Results["ELBO"][i][end]*ones(DiffN)]
        else
            time = tm.Results["Time"][i];
            acc = tm.Results["Accuracy"][i];
            meanl = tm.Results["MeanL"][i];
            medianl = tm.Results["MedianL"][i];
            elbo = tm.Results["ELBO"][i];
        end
        Mtime = hcat(Mtime,time)
        Macc = hcat(Macc,acc)
        Mmeanl = hcat(Mmeanl,meanl)
        Mmedianl = hcat(Mmedianl,medianl)
        Melbo = hcat(Melbo,elbo)
    end
    if size(Mtime,2)!=2
      Mtime[:,2] = Mtime[:,3]
    end
    Mtime = Mtime[:,2:end];  Macc = Macc[:,2:end]
    Mmeanl = Mmeanl[:,2:end]; Mmedianl = Mmedianl[:,2:end]
    Melbo = Melbo[:,2:end];
    tm.Results["Time"] = Mtime;
    tm.Results["Accuracy"] = Macc;
    tm.Results["MeanL"] = Mmeanl
    tm.Results["MedianL"] = Mmedianl
    tm.Results["ELBO"] = Melbo
    tm.Results["Processed"]= [vec(mean(Mtime,2)) vec(std(Mtime,2)) vec(mean(Macc,2)) vec(std(Macc,2)) vec(mean(Mmeanl,2)) vec(std(Mmeanl,2)) vec(mean(Mmedianl,2)) vec(std(Mmedianl,2)) vec(mean(Melbo,2)) vec(std(Melbo,2))]
end

function PrintResults(results,method_name,writing_order)
  println("Model $(method_name) : ")
  i = 1
  for category in writing_order
    println("$category : $(results[i*2-1]) ± $(results[i*2])")
    i+=1
  end
end

function WriteResults(tm::TestingModel,location,writing_order)
  fold = String(location*"/"*tm.ExperimentType*"Experiment"*(doAutotuning ? "_AT":"")*(Experiment==AccuracyExp ?"eps$(tm.Param["ϵ"])":""))
  if !isdir(fold); mkdir(fold); end;
  fold = fold*"/"*tm.DatasetName*"Dataset"
  labels=Array{String,1}(length(writing_order)*2)
  labels[1:2:end-1,:] = writing_order.*"_mean"
  labels[2:2:end,:] =  writing_order.*"_std"
  if !isdir(fold); mkdir(fold); end;
  if Experiment == AccuracyExp
      writedlm(String(fold*"/Results_"*tm.MethodName*".txt"),tm.Results["allresults"])
  else
      writedlm(String(fold*"/Results_"*tm.MethodName*".txt"),tm.Results["Processed"])
  end
end

#Return predicted labels (-1,1) for test set X_test
function ComputePrediction(tm::TestingModel, i,X, X_test)
  y_predic = []
  if typeof(tm.Model[i]) <: OMGP.GPModel
    y_predic = sign.(tm.Model[i].predict(X_test))
  elseif tm.MethodType == "SVGPMC"
    y_predic = sign.(tm.Model[i][:predict_y](X_test)[1]*2-1)
  elseif tm.MethodType == "EPGPMC"
    R"y_predic <- predict($X_test,$(tm.Model[i]))"
    @rget y_predic
    y_predic = sign.(y_predic-0.5)
  elseif tm.MethodType == "TTGPMC"
  elseif tm.MethodType == "ARMPC"
  end
  return y_predic
end

#Return prediction certainty for class 1 on test set X_test
function ComputePredictionAccuracy(tm::TestingModel,i, X, X_test)
  y_predic = []
  if typeof(tm.Model[i]) <: OMGP.GPModel
    y_predic = tm.Model[i].predictproba(X_test)
  elseif tm.MethodType == "SVGPC"
    y_predic = tm.Model[i][:predict_y](X_test)[1]
  elseif tm.MethodType == "LogReg"
    y_predic = tm.Model[i][:predict_proba](X_test)[:,2]
  elseif tm.MethodType == "SVM"
    y_predic = tm.Model[i][:predict_proba](X_test)[:,2]
  elseif tm.MethodType == "EPGPC"
    R"y_predic <- predict($X_test,$(tm.Model[i]))"
    @rget y_predic
 end
  return y_predic
end

#Return Accuracy on test set
function TestAccuracy(y_test, y_predic)
  return 1-sum(1-y_test.*y_predic)/(2*length(y_test))
end
#Return Brier Score
function TestBrierScore(y_test, y_predic)
  return sum(((y_test+1)./2 - y_predic).^2)/length(y_test)
end
#Return Log Score
function TestLogScore(y_test, y_predic)
  return -sum((y_test+1)./2.*log.(y_predic)+(1-(y_test+1)./2).*log.(1-y_predic))/length(y_test)
end

function TestMeanHoldOutLikelihood(loglike)
    return mean(loglike)
end

function TestMedianHoldOutLikelihood(loglike)
    return median(loglike)
end

function HoldOutLikelihood(y_test,y_predic)
    loglike = zeros(y_predic)
    loglike[y_test.==1] = log.(y_predic[y_test.==1])
    loglike[y_test.==-1] = log.(1-y_predic[y_test.==-1])
    return loglike
end


#Return ROC
function ROC(y_test,y_predic,npoints)
    nt = length(y_test)
    truepositive = zeros(npoints); falsepositive = zeros(npoints)
    truenegative = zeros(npoints); falsenegative = zeros(npoints)
    thresh = collect(range(0,1,npoints))
    for i in 1:npoints
      for j in 1:nt
        truepositive[i] += (y_predic[j]>=thresh[i] && y_test[j]>=0.9) ? 1 : 0;
        truenegative[i] += (y_predic[j]<=thresh[i] && y_test[j]<=-0.9) ? 1 : 0;
        falsepositive[i] += (y_predic[j]>=thresh[i] && y_test[j]<=-0.9) ? 1 : 0;
        falsenegative[i] += (y_predic[j]<=thresh[i] && y_test[j]>=0.9) ? 1 : 0;
      end
    end
    return (truepositive./(truepositive+falsenegative),falsepositive./(truenegative+falsepositive))
end

function TestAUCScore(ROC)
    (sensibility,specificity) = ROC
    h = specificity[1:end-1]-specificity[2:end]
    AUC = sum(sensibility[1:end-1].*h)
    return AUC
end

function WriteLastStateParameters(testmodel,top_fold,X_test,y_test,i)
    if isa(testmodel.Model[i],OMGP.GPModel)
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
        if isa(testmodel.Model[i],OMGP.SparseModel)
            writedlm(top_fold*"/ind_points"*"_$i",testmodel.Model[i].inducingPoints)
        end
        if isa(testmodel.Model[i],OMGP.NonLinearModel)
            writedlm(top_fold*"/kernel_param"*"_$i",broadcast(getfield,testmodel.Model[i].kernel,:param))
            writedlm(top_fold*"/kernel_coeff"*"_$i",broadcast(getfield,testmodel.Model[i].kernel,:coeff))
            writedlm(top_fold*"/kernel_name"*"_$i",broadcast(getfield,testmodel.Model[i].kernel,:name))
        end
	println("Last state saved in $top_fold")
    end
end

function PlotResults(TestModels,tests)
    nModels=length(TestModels)
    if nModels == 0; return; end;
    bigwidth = 2
    nResults = length(tests)
    ind = 0.1:bigwidth:(nResults-1)*bigwidth+0.1
    width = bigwidth/(nModels+1)
    figure("Final Results");clf();
    colors=["b", "g", "r", "c", "m", "y", "k", "w"]
    iter=0
    for (name,testmodel) in TestModels
        results = testmodel.Results["allresults"]
        bar(ind+iter*width,results[1:2:end]',width,color=colors[iter+1],yerr=results[2:2:end]',label=name)#
        iter+=1
    end
    xticks(ind+nModels/2*width,tests)
    legend()
end

function PlotResultsConvergence(TestModels)
    nModels=length(TestModels)
    if nModels == 0; return; end;
    figure("Convergence Results");clf();
    colors=["b", "r", "g"]
    subplot(2,2,1); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,3],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,3]-results[1:step:end,4]/sqrt(10),results[1:step:end,3]+results[1:step:end,4]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    ylabel("Accuracy")
    subplot(2,2,2); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,5],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,5]-results[1:step:end,6]/sqrt(10),results[1:step:end,5]+results[1:step:end,6]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    ylabel("Mean Log L")
    subplot(2,2,3); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,7],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,7]-results[1:step:end,8]/sqrt(10),results[1:step:end,7]+results[1:step:end,8]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    ylabel("Median Log L")
    subplot(2,2,4); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,9],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,9]-results[1:step:end,10]/sqrt(10),results[1:step:end,9]+results[1:step:end,10]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    ylabel("neg. ELBO")
    # subplot(3,2,5); #Accuracy
    #     iter=1
    #     step =1
    #     for (name,testmodel) in TestModels
    #         results = testmodel.Results["Processed"]
    #         plot(results[1:step:end,1],results[1:step:end,11],color=colors[iter],label=name)
    #         fill_between(results[1:step:end,1],results[1:step:end,11]-results[1:step:end,12]/sqrt(10),results[1:step:end,11]+results[1:step:end,12]/sqrt(10),alpha=0.2,facecolor=colors[iter])
    #         iter+=1
    #     end
    # legend()
    # xlabel("Time [s]")
    # ylabel("Param")
    # subplot(3,2,6); #Accuracy
    #     iter=1
    #     step =1
    #     for (name,testmodel) in TestModels
    #         results = testmodel.Results["Processed"]
    #         plot(results[1:step:end,1],results[1:step:end,13],color=colors[iter],label=name)
    #         fill_between(results[1:step:end,1],results[1:step:end,13]-results[1:step:end,14]/sqrt(10),results[1:step:end,13]+results[1:step:end,14]/sqrt(10),alpha=0.2,facecolor=colors[iter])
    #         iter+=1
    #     end
    # legend()
    # xlabel("Time [s]")
    # display(ylabel("Coeff"))
end


end #end of module
