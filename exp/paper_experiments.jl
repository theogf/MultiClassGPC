#### Paper_Experiments ####
# Run on a file and compute Accuracy on a nFold cross validation
# Compute also the brier score and the logscore

include("get_arguments.jl")

# if !in(LOAD_PATH,SRC_PATH); push!(LOAD_PATH,SRC_PATH); end;
#Compare XGPMC, BSVM, SVGPMC and others

#Methods and scores to test
doSXGPMC = args["XGP"] #Sparse XGPMC (sparsity)
doEPGPMC = args["EPGP"]
doSVGPMC = args["SVGP"] #Sparse Variational GPMC (Hensmann)
doARMC = args["AR"]
doTTGPMC = args["TTGP"]

doBXGPMC = false
doStochastic = args["stochastic"]
doAutotuning = args["autotuning"]
doPointOptimization = args["point-optimization"]

include("functions_paper_experiment.jl")

ExperimentName = "Convergence"
doSaveLastState = args["last-state"]
doPlot = args["plot"]
if doPlot
    using PyPlot
end
doWrite = !args["no-writing"] #Write results in approprate folder
ShowIntResults = true #Show intermediate time, and results for each fold


#Testing Parameters
#= Datasets available are X :
aXa, Bank_marketing, Click_Prediction, Cod-rna, Diabetis, Electricity, German, Shuttle
=#
# dataset = "isolet"
dataset = args["dataset"]
(X_data,y_data,DatasetName) = get_Dataset(dataset)
MaxIter = args["maxiter"] #Maximum number of iterations for every algorithm
iter_points= vcat(1:9,10:5:99,100:50:999,1000:1000:9999)
(nSamples,nFeatures) = size(X_data);
nFold = args["nFold"]; #Choose the number of folds
iFold = args["iFold"] > nFold ? nFold : args["iFold"]; #Number of fold to estimate
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold

#Main Parameters
main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-10 #Convergence criterium
main_param["maxIter"]=MaxIter
main_param["M"] = args["indpoints"]!=0 ? args["indpoints"] : min(100,floor(Int64,0.2*nSamples)) #Number of inducing points
main_param["Kernel"] = "rbf"
l = initial_lengthscale(X_data)
main_param["Θ"] = 0.5 #initial Hyperparameter of the kernel
main_param["var"] = 10.0 #Variance
main_param["nClasses"] = length(unique(y_data))
main_param["BatchSize"] = args["batchsize"]
main_param["Verbose"] = 1
main_param["Window"] = 10
main_param["Autotuning"] = doAutotuning
main_param["PointOptimization"] = doPointOptimization
#All Parameters
BXGPMCParam = XGPMCParameters(main_param=main_param,independent=true)
SXGPMCParam = XGPMCParameters(Stochastic=doStochastic,Sparse=true,ALR=true,main_param=main_param,independent=true)
SVGPMCParam = SVGPMCParameters(Stochastic=doStochastic,main_param=main_param)
EPGPMCParam = EPGPMCParameters(Stochastic=doStochastic,main_param=main_param)


#Set of all models
TestModels = Dict{String,TestingModel}()

if doBXGPMC; TestModels["BXGPMC"] = TestingModel("BXGPMC",DatasetName,ExperimentName,"BXGPMC",BXGPMCParam); end;
if doSXGPMC; TestModels["SXGPMC"] = TestingModel("SXGPMC",DatasetName,ExperimentName,"SXGPMC",SXGPMCParam); end;
if doSVGPMC;   TestModels["SVGPMC"]   = TestingModel("SVGPMC",DatasetName,ExperimentName,"SVGPMC",SVGPMCParam);      end;
if doEPGPMC; TestModels["EPGPMC"] = TestingModel("EPGPMC",DatasetName,ExperimentName,"EPGPMC",EPGPMCParam); end;

writing_order = ["Time","Accuracy","MeanL","MedianL","ELBO"]

#Main printing
print("Dataset $dataset loaded, starting $ExperimentName experiment,")
print(" with $iFold fold out of $nFold, autotuning = $doAutotuning and optindpoints = $doPointOptimization,")
print(" max of $MaxIter iterations\n")

for (name,testmodel) in TestModels
    println("Running $(testmodel.MethodName) on $(testmodel.DatasetName) dataset")
    #Initialize the results storage
    testmodel.Model = Array{Any}(undef,nFold)
    testmodel.Results["Time"] = Vector{Vector{Float64}}(undef,nFold);
    testmodel.Results["Accuracy"] = Vector{Vector{Float64}}(undef,nFold);
    testmodel.Results["MeanL"] = Vector{Vector{Float64}}(undef,nFold);
    testmodel.Results["MedianL"] = Vector{Vector{Float64}}(undef,nFold);
    testmodel.Results["ELBO"] = Vector{Vector{Float64}}(undef,nFold);
    for i in 1:iFold #Run over iFold folds of the data
        if ShowIntResults
            println("#### Fold number $i/$nFold ####")
        end
        X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
        y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
        if (length(y_test) > 10000 )#When test set is too big, reduce it for time purposes
            subset = StatsBase.sample(1:length(y_test),10000,replace=false)
            X_test = X_test[subset,:];
            y_test = y_test[subset];
        end
        X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
        y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
        init_t = time_ns()
        CreateModel!(testmodel,i,X,y)
        if testmodel.MethodType == "EPGPMC"
            global LogArrays=copy(transpose(TrainModel!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)))
            testmodel.Results["Time"][i] = LogArrays[1,:]
        else
            global LogArrays= hcat(TrainModel!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)...)
            a = TreatTime(init_t,LogArrays[1,:],LogArrays[6,:])
            testmodel.Results["Time"][i] = a
        end
        testmodel.Results["Accuracy"][i] = LogArrays[2,:]
        testmodel.Results["MeanL"][i] = LogArrays[3,:]
        testmodel.Results["MedianL"][i] = LogArrays[4,:]
        testmodel.Results["ELBO"][i] = LogArrays[5,:]
        if ShowIntResults
            println("$(testmodel.MethodName) : Time  = $((time_ns()-init_t)*1e-9)s, accuracy : $(LogArrays[2,end])")
        end
        if doWrite && doSaveLastState
            top_fold = "results";
            if !isdir(top_fold); mkdir(top_fold); end;
            WriteLastStateParameters(testmodel,top_fold,X_test,y_test,i)
        end
        #Reset the kernel
        if testmodel.MethodName == "SVGPMC"
            testmodel.Param["Kernel"] = gpflow.kernels[:Sum]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=false),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
        elseif testmodel.MethodName == "SXGPMC"
            println("SXGPMC : params : $(OMGP.getvalue(testmodel.Model[i].kernel[1].param[1])), and coeffs $(OMGP.getvalue(testmodel.Model[i].kernel[1].variance))")
        end
    end # of the loop over the folds
    ProcessResults(testmodel,iFold)
    println(size(testmodel.Results["Processed"]))
    if doWrite
        top_fold = "results";
        if !isdir(top_fold); mkdir(top_fold); end;
        WriteResults(testmodel,top_fold,writing_order) #Write the results in an adapted format into a folder
    end
end #Loop over the models
if doPlot
    PlotResults(TestModels)
end
