#### Paper_Experiments ####
# Run on a file and compute Accuracy on a nFold cross validation
# Compute also the brier score and the logscore

include("get_arguments.jl")

#Add the path necessary modules
PWD = pwd()
if isdir(PWD*"/src")
    SRC_PATH = pwd()*"/src"
else
    SRC_PATH = pwd()*"/../src"
end
if !in(LOAD_PATH,SRC_PATH); push!(LOAD_PATH,SRC_PATH); end;
include("functions_paper_experiment.jl")

#Compare XGPMC, BSVM, SVGPMC and others

#Methods and scores to test
doSXGPMC = args["XGP"] #Sparse XGPMC (sparsity)
doEPGPMC = args["EPGP"]
doSVGPMC = args["SVGP"] #Sparse Variational GPMC (Hensmann)
doARMC = args["AR"]
doTTGPMC = args["TTGP"]

doAutotuning = true#!!!args["autotuning"]
doPointOptimization = args["point-optimization"]

# ExperimentName = "Prediction"
ExperimentName = args["exp"]
ExperimentTypes = Dict("Convergence"=>ConvergenceExp,"Accuracy"=>AccuracyExp)
doTime = true #Return time needed for training
doAccuracy = true #Return Accuracy
doBrierScore = true # Return BrierScore
doLogScore = true #Return LogScore
doAUCScore = true
doLikelihoodScore = true
doSaveLastState = args["last-state"]
doPlot = args["plot"]
doWrite = !args["no-writing"] #Write results in approprate folder
ShowIntResults = true #Show intermediate time, and results for each fold


#Testing Parameters
#= Datasets available are X :
aXa, Bank_marketing, Click_Prediction, Cod-rna, Diabetis, Electricity, German, Shuttle
=#
dataset = "Click_Prediction"#!!!args["dataset"]
(X_data,y_data,DatasetName) = get_Dataset(dataset)
MaxIter = 100#!!!args["maxiter"] #Maximum number of iterations for every algorithm
iter_points= vcat(1:99,100:10:999,1000:1000:9999)
(nSamples,nFeatures) = size(X_data);
nFold = args["nFold"]; #Choose the number of folds
iFold = 3;#!!!args["iFold"] > nFold ? nFold : args["iFold"]; #Number of fold to estimate
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold

#Main Parameters
main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-10 #Convergence criterium
main_param["maxIter"]=MaxIter
main_param["M"] = args["indpoints"]!=0 ? args["indpoints"] : min(100,floor(Int64,0.2*nSamples)) #Number of inducing points
main_param["Kernel"] = "rbf"
main_param["Θ"] = 1.5 #initial Hyperparameter of the kernel
main_param["BatchSize"] = args["batchsize"]
main_param["Verbose"] = 1
main_param["Window"] = 10
main_param["Autotuning"] = doAutotuning
main_param["PointOptimization"] = doPointOptimization
#All Parameters
BXGPMCParam = XGPMCParameters(main_param=main_param)
SXGPMCParam = XGPMCParameters(Stochastic=true,Sparse=true,ALR=true,main_param=main_param)
SVGPMCParam = SVGPMCParameters(Stochastic=true,Sparse=true,main_param=main_param)
EPGPMCParam = EPGPMCParameters(main_param=main_param)


#Set of all models
TestModels = Dict{String,TestingModel}()

if doBXGPMC; TestModels["BXGPMC"] = TestingModel("BXGPMC",DatasetName,ExperimentName,"BXGPMC",BXGPMCParam); end;
if doSXGPMC; TestModels["SXGPMC"] = TestingModel("SXGPMC",DatasetName,ExperimentName,"SXGPMC",SXGPMCParam); end;
if doSVGPMC;   TestModels["SVGPMC"]   = TestingModel("SVGPMC",DatasetName,ExperimentName,"SVGPMC",SVGPMCParam);      end;
if doEPGPMC; TestModels["EPGPMC"] = TestingModel("EPGPMC",DatasetName,ExperimentName,"EPGPMC",EPGPMCParam); end;

writing_order = Array{String,1}();                    if doTime; push!(writing_order,"Time"); end;
if doAccuracy; push!(writing_order,"Accuracy"); end;  if doBrierScore; push!(writing_order,"Brierscore"); end;
if doLogScore; push!(writing_order,"-Logscore"); end;  if doAUCScore; push!(writing_order,"AUCscore"); end;
if doLikelihoodScore; push!(writing_order,"-MedianL"); push!(writing_order,"-MeanL"); end;

#Main printing
print("Dataset $dataset loaded, starting $ExperimentName experiment,")
print(" with $iFold fold out of $nFold, autotuning = $doAutotuning and optindpoints = $doPointOptimization,")
print(" max of $MaxIter iterations\n")

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
        if doTime;        testmodel.Results["Time"]       = Array{Float64,1}(nFold);end;
        if doAccuracy;    testmodel.Results["Accuracy"]   = Array{Float64,1}(nFold);end;
        if doBrierScore;  testmodel.Results["Brierscore"] = Array{Float64,1}(nFold);end;
        if doLogScore;    testmodel.Results["-Logscore"]   = Array{Float64,1}(nFold);end;
        if doAUCScore;    testmodel.Results["AUCscore"]   = Array{Float64,1}(nFold);end;
        if doLikelihoodScore;  testmodel.Results["-MedianL"] = Array{Float64,1}(nFold);
                            testmodel.Results["-MeanL"] = Array{Float64,1}(nFold);end;
    end #of initialization
    for i in 1:iFold #Run over iFold folds of the data
        if ShowIntResults
            println("#### Fold number $i/$nFold ####")
        end
        X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
        y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
        if (length(y_test) > 100000 )#When test set is too big, reduce it for time purposes
            subset = StatsBase.sample(1:length(y_test),100000,replace=false)
            X_test = X_test[subset,:];
            y_test = y_test[subset];
        end
        X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
        y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
        init_t = time_ns()
        CreateModel!(testmodel,i,X,y)
        if Experiment == AccuracyExp
            TrainModel!(testmodel,i,X,y,X_test,y_test,MaxIter)
            tot_time = (time_ns()-init_t)*1e-9
            if doTime; testmodel.Results["Time"][i] = tot_time; end;
            RunTests(testmodel,i,X,X_test,y_test,Accuracy=doAccuracy,brierscore=doBrierScore,logscore=doLogScore,AUCscore=doAUCScore,likelihoodscore=doLikelihoodScore)
        elseif Experiment == ConvergenceExp
            if testmodel.MethodType == "LogReg"
                TrainModelwithTime!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)
            else
                LogArrays= hcat(TrainModelwithTime!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)...)
                if testmodel.MethodType == "EPGPMC"
                    LogArrays=LogArrays'
                    testmodel.Results["Time"][i] = LogArrays[1,:]
                else
                    testmodel.Results["Time"][i] = TreatTime(init_t,LogArrays[1,:],LogArrays[6,:])
                end
                testmodel.Results["Accuracy"][i] = LogArrays[2,:]
                testmodel.Results["MeanL"][i] = LogArrays[3,:]
                testmodel.Results["MedianL"][i] = LogArrays[4,:]
                testmodel.Results["ELBO"][i] = LogArrays[5,:]
                #testmodel.Results["Param"][i] = LogArrays[7,:]
                #testmodel.Results["Coeff"][i] = LogArrays[8,:]
            end
        end #Training in function of the type of experiment
        if ShowIntResults
            println("$(testmodel.MethodName) : Time  = $((time_ns()-init_t)*1e-9)s, accuracy : $(LogArrays[2,end])")
        end
        if doWrite && doSaveLastState
            top_fold = SRC_PATH*"/../results";
            if !isdir(top_fold); mkdir(top_fold); end;
            WriteLastStateParameters(testmodel,top_fold,X_test,y_test,i)
        end
        #Reset the kernel
        if testmodel.MethodName == "SVGPMC"
            # rbf = testmodel.Model[i][:kern][:kernels][1]
            # println("SVGPMC : params : $(rbf[:lengthscales][:value]) and coeffs : $(rbf[:variance][:value])")
            testmodel.Param["Kernel"] = gpflow.kernels[:Sum]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=false),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
        elseif testmodel.MethodName == "SXGPMC"
            println("SXGPMC : params : $(OMGP.getvalue(testmodel.Model[i].kernel.param[1])), and coeffs $(OMGP.getvalue(testmodel.Model[i].kernel.weight))")
        end
    end # of the loop over the folds
    if Experiment == AccuracyExp
        ProcessResults(testmodel,writing_order) #Compute mean and std deviation
        PrintResults(testmodel.Results["allresults"],testmodel.MethodName,writing_order) #Print the Results in the end
    else
        ProcessResultsConvergence(testmodel,iFold)
        println(size(testmodel.Results["Processed"]))
    end
    if doWrite
        top_fold = "results";
        if !isdir(top_fold); mkdir(top_fold); end;
        WriteResults(testmodel,top_fold,writing_order) #Write the results in an adapted format into a folder
    end
end #Loop over the models
if doPlot
    if Experiment != ConvergenceExp
        PlotResults(TestModels,writing_order)
    else
        PlotResultsConvergence(TestModels)
    end
end

if doPlot
    println("End of analysis, press enter to exit")
    readline(STDIN);
end
