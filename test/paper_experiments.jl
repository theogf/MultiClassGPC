#### Paper_Experiment_Predictions ####
# Run on a file and compute accuracy on a nFold cross validation
# Compute also the brier score and the logscore

push!(LOAD_PATH,".")
if !isdefined(:TestFunctions); include("paper_experiment_functions.jl");end;
# using TestFunctions
using DataAccess
#Compare SEP, SVGP, A&R, TT and X-MGPC

#Methods and scores to test
doXMGPC = true
doSEP = false
doTT = false
doSVGP = false
doMultiNomial = false

ExperimentName = "Prediction"
# ExperimentName = "ConvergenceExperiment"
@enum ExperimentType PredictionExp=1 ConvergenceExp=2
ExperimentTypes = Dict("ConvergenceExperiment"=>ConvergenceExp,"Prediction"=>PredictionExp)
Experiment = ExperimentTypes[ExperimentName]
doTime = true #Return time needed for training
doAccuracy = true #Return Accuracy
doBrierScore = true # Return BrierScore
doLogScore = true #Return LogScore
doAUCScore = true
doLikelihoodScore = true
doPlot = false
doWrite = true #Write results in approprate folder
ShowIntResults = true #Show intermediate time, and results for each fold

#Testing Parameters
#= Datasets available are get_X :
Ionosphere,Sonar,Crabs,USPS, Banana, Image, RingNorm
BreastCancer, Titanic, Splice, Diabetis, Thyroid, Heart, Waveform, Flare
=#
ARGS = ("German","10")
dataset = ARGS[1]
 # (X_data,y_data,DatasetName) = get_BreastCancer()
(X_data,y_data,DatasetName) = get_Dataset(dataset)
MaxIter = 1000 #Maximum number of iterations for every algorithm
#iter_points = [1:1:9]#;10:10:99;100:100:999;1000:100:9999;10000:1000:99999;100000:10000:1000000]
(nSamples,nFeatures) = size(X_data);
nFold = 10; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold

#Main Parameters
main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-10 #Convergence criterium
main_param["M"] = parse(Int64,ARGS[2]) #min(100,floor(Int64,0.2*nSamples))
main_param["Kernel"] = "rbf"
main_param["Θ"] = 6.0 #Hyperparameter of the kernel
main_param["BatchSize"] = 100
main_param["Verbose"] = 0
main_param["Window"] = 10
main_param["Storing"] = false
#BSVM and GPC Parameters
BXGPCParam = XGPCParameters(Stochastic=false,Sparse=false,ALR=true,main_param=main_param)
SXGPCParam = XGPCParameters(Stochastic=true,Sparse=true,ALR=true,main_param=main_param)
LBSVMParam = BSVMParameters(Stochastic=false,NonLinear=true)
BBSVMParam = BSVMParameters(Stochastic=false,Sparse=false,ALR=false,main_param=main_param)
SBSVMParam = BSVMParameters(Stochastic=true,Sparse=true,ALR=true,main_param=main_param)
LogRegParam = LogRegParameters(main_param=main_param)
GPCParam = GPCParameters(Stochastic=true,Sparse=true,main_param=main_param)
ECMParam = ECMParameters(main_param=main_param)
SVMParam = SVMParameters(main_param=main_param)

#SXGPCParam["Autotuning"] = true; SXGPCParam["ρ"] = 0.01;
#SXGPCParam["ATFrequency"] = 20;
# BBSVMParam["Autotuning"] = false; BBSVMParam["ATFrequency"] = 2

#Global variables for debugging
#X = []; y = []; X_test = []; y_test = [];

#Set of all models
TestModels = Dict{String,TestingModel}()

if doBXGPC; TestModels["BXGPC"] = TestingModel("BXGPC",DatasetName,ExperimentName,"BXGPC",BXGPCParam); end;
if doSXGPC; TestModels["SXGPC"] = TestingModel("SXGPC",DatasetName,ExperimentName,"SXGPC",SXGPCParam); end;
if doLBSVM; TestModels["LBSVM"] = TestingModel("LBSVM",DatasetName,ExperimentName,"LBSVM",BBSVMParam); end;
if doBBSVM; TestModels["BBSVM"] = TestingModel("BBSVM",DatasetName,ExperimentName,"BBSVM",BBSVMParam); end;
if doSBSVM; TestModels["SBSVM"] = TestingModel("SBSVM",DatasetName,ExperimentName,"SBSVM",SBSVMParam); end;
if doLogReg; TestModels["LogReg"] = TestingModel("LogReg",DatasetName,ExperimentName,"LogReg",LogRegParam); end;
if doPlatt; TestModels["Platt"] = TestingModel("SVM",DatasetName,ExperimentName,"SVM",SVMParam);      end;
if doGPC;   TestModels["GPC"]   = TestingModel("GPC",DatasetName,ExperimentName,"GPC",GPCParam);      end;
if doECM;   TestModels["ECM"]   = TestingModel("ECM",DatasetName,ExperimentName,"ECM",ECMParam);      end;

writing_order = Array{String,1}();                    if doTime; push!(writing_order,"time"); end;
if doAccuracy; push!(writing_order,"accuracy"); end;  if doBrierScore; push!(writing_order,"brierscore"); end;
if doLogScore; push!(writing_order,"-logscore"); end;  if doAUCScore; push!(writing_order,"AUCscore"); end;
if doLikelihoodScore; push!(writing_order,"medianlikelihoodscore"); push!(writing_order,"meanlikelihoodscore"); end;
#conv_BSVM = falses(nFold); conv_SBSVM = falses(nFold); conv_SSBSVM = falses(nFold); conv_GPC = falses(nFold); conv_SGPC = falses(nFold); conv_SSGPC = falses(nFold); conv_EM = falses(nFold); conv_FITCEM = falses(nFold); conv_SVM = falses(nFold)
for (name,testmodel) in TestModels
  println("Running $(testmodel.MethodName) on $(testmodel.DatasetName) dataset")
  #Initialize the results storage
  testmodel.Model = Array{Any}(nFold)
  if Experiment == ConvergenceExp
      testmodel.Results["Time"] = Array{Any}(nFold);
      testmodel.Results["Accuracy"] = Array{Any}(nFold);
      testmodel.Results["MeanL"] = Array{Any}(nFold);
      testmodel.Results["MedianL"] = Array{Any}(nFold);
      testmodel.Results["ELBO"] = Array{Any}(nFold);
  else
      if doTime;        testmodel.Results["time"]       = Array{Float64,1}(nFold);end;
      if doAccuracy;    testmodel.Results["accuracy"]   = Array{Float64,1}(nFold);end;
      if doBrierScore;  testmodel.Results["brierscore"] = Array{Float64,1}(nFold);end;
      if doLogScore;    testmodel.Results["-logscore"]   = Array{Float64,1}(nFold);end;
      if doAUCScore;    testmodel.Results["AUCscore"]   = Array{Float64,1}(nFold);end;
      if doLikelihoodScore;  testmodel.Results["medianlikelihoodscore"] = Array{Float64,1}(nFold);
                        testmodel.Results["meanlikelihoodscore"] = Array{Float64,1}(nFold);end;
  end
 #Threads.@threads
 for i = 1:nFold #Run over all folds of the data
    if ShowIntResults
#      println("#### Fold number $i/$nFold processed by thread $(Threads.threadid())###")
#      println("#### Fold number $i/$nFold###")
    end
    X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
    y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
    if (length(y_test) > 100000 )
        subset = StatsBase.sample(1:length(y_test),100000,replace=false)
        X_test = X_test[subset,:];
        y_test = y_test[subset];
    end
    X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
    y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
    init_t = time_ns()
    CreateModel!(testmodel,i,X,y)
    if Experiment == PredictionExp || Experiment == FrameworkExp
        TrainModel!(testmodel,i,X,y,X_test,y_test,MaxIter)
        tot_time = (time_ns()-init_t)*1e-9
        if doTime; testmodel.Results["time"][i] = tot_time; end;
        RunTests(testmodel,i,X,X_test,y_test,accuracy=doAccuracy,brierscore=doBrierScore,logscore=doLogScore,AUCscore=doAUCScore,likelihoodscore=doLikelihoodScore)
    elseif Experiment == ConvergenceExp
        LogArrays= hcat(TrainModelwithTime!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)...)
        testmodel.Results["Time"][i] = TreatTime(init_t,LogArrays[1,:],LogArrays[6,:])
        testmodel.Results["Accuracy"][i] = LogArrays[2,:]
        testmodel.Results["MeanL"][i] = LogArrays[3,:]
        testmodel.Results["MedianL"][i] = LogArrays[4,:]
        testmodel.Results["ELBO"][i] = LogArrays[5,:]
    end
    if ShowIntResults
    #   println("$(testmodel.MethodName) : Time  = $(logt[end])")
    end
  end
  # wait(process)
  if Experiment != ConvergenceExp
      ProcessResults(testmodel,writing_order) #Compute mean and std deviation
      PrintResults(testmodel.Results["allresults"],testmodel.MethodName,writing_order) #Print the Results in the end
  else
     # ProcessResultsConvergence(testmodel)
     res = testmodel.Results; l = length(res["Time"][1])
     testmodel.Results["Processed"] = [res["Time"][1] zeros(l) res["Accuracy"][1] zeros(l) res["MeanL"][1] zeros(l) res["MedianL"][1] zeros(l) res["ELBO"][1] zeros(l)]
      println(size(testmodel.Results["Processed"]))
  end
  if doWrite
    top_fold = "data";
    if !isdir(top_fold); mkdir(top_fold); end;
    WriteResults(testmodel,top_fold,writing_order) #Write the results in an adapted format into a folder
  end
end
if doPlot
    if Experiment != ConvergenceExp
        PlotResults(TestModels,writing_order)
    else
        PlotResultsConvergence(TestModels)
    end
end
# end
