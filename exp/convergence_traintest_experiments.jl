#### Paper_Experiments ####
# Run on a file and compute Accuracy on a nFold cross validation
# Compute also the brier score and the logscore

include("get_arguments.jl")
cd(@__DIR__)
# if !in(LOAD_PATH,SRC_PATH); push!(LOAD_PATH,SRC_PATH); end;
#Compare SCGPMC, BSVM, SVGPMC and others

#Methods and scores to test
doSCGPMC = args["SCGP"] #Sparse SCGPMC (sparsity)
doHSCGPMC = args["HSCGP"]
doEPGPMC = args["EPGP"]
doSVGPMC = args["SVGP"] #Sparse Variational GPMC (Hensmann)
doARMC = args["AR"]
doTTGPMC = args["TTGP"]

doBCGPMC = false
doStochastic = args["stochastic"]
doAutotuning = args["autotuning"]
doPointOptimization = args["point-optimization"]
doIndependent = args["independent"]

include("functions_paper_experiment.jl")
include("metrics.jl")
ExperimentName = "Convergence"
doSaveLastState = args["last-state"]
doPlot = args["plot"]
doWrite = !args["no-writing"] #Write results in approprate folder
ShowIntResults = true #Show intermediate time, and results for each fold


#Testing Parameters
#= Datasets available are X :
aXa, Bank_marketing, Click_Prediction, Cod-rna, Diabetis, Electricity, German, Shuttle
=#
# dataset = "segment"
dataset = args["dataset"]
X_train,y_train,X_test,y_test,DatasetName = get_train_test(dataset)
MaxIter = args["maxiter"] #Maximum number of iterations for every algorithm
iter_points= vcat(1:9,10:5:99,100:50:999,1e3:500:(1e4-1),1e4:5000:1e5)

(nSamples,nFeatures) = size(X_train);
nFold = 1;#args["nFold"]; #Choose the number of folds
iFold = 1;#args["iFold"] > nFold ? nFold : args["iFold"]; #Number of fold to estimate
N_test_max = 10000
subset = 1:length(y_test)
if length(y_test) > N_test_max
    subset = StatsBase.sample(1:length(y_test),N_test_max,replace=false)
end
X_test = X_test[subset,:]
y_test = y_test[subset]
#Main Parameters
main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-10 #Convergence criterium
main_param["maxIter"]=MaxIter
main_param["γ"] = 0.0
main_param["M"] = args["indpoints"]!=0 ? args["indpoints"] : min(100,floor(Int64,0.2*nSamples)) #Number of inducing points
main_param["Kernel"] = "ard"
# main_param["Kernel"] = "iso"
# main_param["Kernel"] = "ard"
l = initial_lengthscale(X_train)
main_param["Θ"] = sqrt(l) #initial Hyperparameter of the kernel
main_param["var"] = 1.0 #Variance
main_param["nClasses"] = length(unique(y_train))
main_param["BatchSize"] = args["batchsize"]
main_param["Verbose"] = 0
main_param["Window"] = 10
main_param["Autotuning"] = doAutotuning
main_param["PointOptimization"] = doPointOptimization
main_param["independent"] = doIndependent
#All Parameters
BCGPMCParam = CGPMCParameters(main_param=main_param)
# SCGPMCParam = CGPMCParameters(Stochastic=doStochastic,Sparse=true,optimizer=VanillaGradDescent(η=1.0),main_param=main_param)
SCGPMCParam = CGPMCParameters(Stochastic=doStochastic,Sparse=true,optimizer=InverseDecay(τ=0),main_param=main_param)
# SCGPMCParam = CGPMCParameters(Stochastic=doStochastic,Sparse=true,optimizer=ALRSVI(),main_param=main_param)
HSCGPMCParam = CGPMCParameters(dohybrid=true,Stochastic=doStochastic,Sparse=true,main_param=main_param)
SVGPMCParam = SVGPMCParameters(Stochastic=doStochastic,main_param=main_param)
EPGPMCParam = EPGPMCParameters(Stochastic=doStochastic,main_param=main_param)


#Set of all models
TestModels = Dict{String,TestingModel}()

if doBCGPMC; TestModels["BCGPMC"] = TestingModel("BCGPMC",DatasetName,ExperimentName,"BCGPMC",BCGPMCParam); end;
if doSCGPMC; TestModels["SCGPMC"] = TestingModel("SCGPMC",DatasetName,ExperimentName,"SCGPMC",SCGPMCParam); end;
if doHSCGPMC; TestModels["HSCGPMC"] = TestingModel("HSCGPMC",DatasetName,ExperimentName,"SCGPMC",HSCGPMCParam); end;
if doSVGPMC;   TestModels["SVGPMC"]   = TestingModel("SVGPMC",DatasetName,ExperimentName,"SVGPMC",SVGPMCParam);      end;
if doEPGPMC; TestModels["EPGPMC"] = TestingModel("EPGPMC",DatasetName,ExperimentName,"EPGPMC",EPGPMCParam); end;

writing_order = ["Time","Accuracy","MeanL","MedianL","ELBO"]

#Main printing
print("Dataset $dataset loaded, starting $ExperimentName experiment,")
print(" with $iFold fold out of $nFold, autotuning = $doAutotuning and optindpoints = $doPointOptimization,")
print(" max of $MaxIter iterations\n")

for (name,testmodel) in TestModels
    println("Running $(testmodel.MethodName) on $(testmodel.DatasetName) dataset")
    #Removign initialization overhead
    testmodel.Model = Array{Any}(undef,1)
    if testmodel.MethodType == "SCGPMC"
        CreateModel!(testmodel,1,X_train,y_train)
        TrainModel!(testmodel,1,X_train,y_train,X_test,y_test,20,0)
        println("Overhead removed")
    end
    #Initialize the results storage
    testmodel.Results["Time"] = Vector{Float64}();
    testmodel.Results["Accuracy"] = Vector{Float64}();
    testmodel.Results["MeanL"] = Vector{Float64}();
    testmodel.Results["MedianL"] = Vector{Float64}();
    testmodel.Results["ELBO"] = Vector{Float64}();
    testmodel.Results["AUC"] = Vector{Float64}();
    testmodel.Results["ECE"] = Vector{Float64}();
    testmodel.Results["MCE"] = Vector{Float64}();
    CreateModel!(testmodel,1,X_train,y_train)
    init_t = time_ns()
    if testmodel.MethodType == "EPGPMC"
        global LogArrays=copy(transpose(TrainModel!(testmodel,1,X_train,y_train,X_test,y_test,MaxIter,iter_points)))
        testmodel.Results["Time"] = LogArrays[1,:]
        testmodel.Results["AUC"] = LogArrays[6,:]
        testmodel.Results["ECE"] = LogArrays[7,:]
        testmodel.Results["MCE"] = LogArrays[8,:]
    else
        global LogArrays= hcat(TrainModel!(testmodel,1,X_train,y_train,X_test,y_test,MaxIter,iter_points)...)
        testmodel.Results["Time"] = TreatTime(init_t,LogArrays[1,:],LogArrays[6,:])
        testmodel.Results["AUC"] = LogArrays[7,:]
        testmodel.Results["ECE"] = LogArrays[8,:]
        testmodel.Results["MCE"] = LogArrays[9,:]
    end
    testmodel.Results["Accuracy"] = LogArrays[2,:]
    testmodel.Results["MeanL"] = LogArrays[3,:]
    testmodel.Results["MedianL"] = LogArrays[4,:]
    testmodel.Results["ELBO"] = LogArrays[5,:]

    if ShowIntResults
        println("$(testmodel.MethodName) : Time  = $((time_ns()-init_t)*1e-9)s, accuracy : $(LogArrays[2,end])")
    end
    #Reset the kernel
    if testmodel.MethodName == "SVGPMC"
        testmodel.Param["Kernel"] = gpflow[:kernels][:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=main_param["Kernel"] == "ard")
        println("SVGPC : params : l =  $(testmodel.Model[1][:kern][:lengthscales][:value]), var = $(testmodel.Model[1][:kern][:variance][:value])")
    elseif testmodel.MethodName == "SCGPMC"
        println("SCGPMC : params : $([AugmentedGaussianProcesses.KernelModule.getlengthscales(k) for k in testmodel.Model[1].kernel])\n and coeffs :  $([AugmentedGaussianProcesses.KernelModule.getvariance(k) for k in testmodel.Model[1].kernel])")
    end
    n = size(testmodel.Results["Time"])
    testmodel.Results["Processed"]= [testmodel.Results["Time"] zeros(n) testmodel.Results["Accuracy"] zeros(n) testmodel.Results["MeanL"] zeros(n) testmodel.Results["MedianL"] zeros(n) testmodel.Results["ELBO"] zeros(n) testmodel.Results["AUC"] zeros(n) testmodel.Results["ECE"] zeros(n) testmodel.Results["MCE"] zeros(n)]
    if doWrite
        top_fold = "results";
        if !isdir(top_fold); mkdir(top_fold); end;
        WriteLastProba(testmodel,top_fold,X_test,y_test)
        WriteResults(testmodel,top_fold,writing_order,false) #Write the results in an adapted format into a folder
    end
end #Loop over the models
if doPlot
    using Plots
    pyplot()
    PlotResults(TestModels)
end
