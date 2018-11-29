#### Paper_Experiments ####
# Run on a file and compute Accuracy on a nFold cross validation
# Compute also the brier score and the logscore

include("get_arguments.jl")

# if !in(LOAD_PATH,SRC_PATH); push!(LOAD_PATH,SRC_PATH); end;
#Compare XGPMC, BSVM, SVGPMC and others

#Methods and scores to test
doSXGPMC = true #Sparse XGPMC (sparsity)
doSVGPMC = true #Sparse Variational GPMC (Hensmann)

doBXGPMC = false
doStochastic = true
doAutotuning = true
doPointOptimization = false

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
# dataset = "mnist"
dataset = args["dataset"]
(X_data,y_data,DatasetName) = get_Dataset(dataset)
MaxIter = 10 #Maximum number of iterations for every algorithm
iter_points= vcat(1:9,10:5:99,100:50:999,1e3:1e3:(1e4-1),1e4:1e4:1e5)

(nSamples,nFeatures) = size(X_data);
nFold = args["nFold"]; #Choose the number of folds
iFold = args["iFold"] > nFold ? nFold : args["iFold"]; #Number of fold to estimate
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
N_test_max = 10000
if nSamples/nFold > N_test_max
        subset = StatsBase.sample(1:floor(Int64,nSamples/nFold),N_test_max,replace=false)
end

#Main Parameters
main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-10 #Convergence criterium
main_param["maxIter"]=MaxIter
main_param["γ"] = 1e-3
main_param["M"] = 200 #Number of inducing points
main_param["Kernel"] = "ard"
l = initial_lengthscale(X_data)
main_param["Θ"] = sqrt(l) #initial Hyperparameter of the kernel
main_param["var"] = 1.0 #Variance
main_param["nClasses"] = length(unique(y_data))
main_param["BatchSize"] = args["batchsize"]
main_param["Verbose"] = 1
main_param["Window"] = 10
main_param["Autotuning"] = doAutotuning
main_param["PointOptimization"] = doPointOptimization
#All Parameters
SXGPMCParam = XGPMCParameters(Stochastic=doStochastic,Sparse=true,ALR=true,main_param=main_param,independent=true)
SVGPMCParam = SVGPMCParameters(Stochastic=doStochastic,main_param=main_param)


#Set of all models
TestModels = Dict{String,TestingModel}()

if doSXGPMC; TestModels["SXGPMC"] = TestingModel("SXGPMC",DatasetName,ExperimentName,"SXGPMC",SXGPMCParam); end;
if doSVGPMC;   TestModels["SVGPMC"]   = TestingModel("SVGPMC",DatasetName,ExperimentName,"SVGPMC",SVGPMCParam);      end;

writing_order = ["Time","Accuracy","MeanL","MedianL","ELBO"]

#Main printing
print("Dataset $dataset loaded, starting $ExperimentName experiment,")
print(" with $iFold fold out of $nFold, autotuning = $doAutotuning and optindpoints = $doPointOptimization,")
print(" max of $MaxIter iterations\n")

for (name,testmodel) in TestModels
    println("Running $(testmodel.MethodName) on $(testmodel.DatasetName) dataset")
    tmod = zeros(iFold)
    testmodel.Model = Array{Any}(undef,nFold)
    for i in 1:iFold #Run over iFold folds of the data
        if ShowIntResults
            println("#### Fold number $i/$nFold ####")
        end
        X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
        y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
        if nSamples/nFold > N_test_max
            X_test = X_test[subset,:]
            y_test = y_test[subset,:]
        end
        X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
        y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
        init_t = time_ns()
        CreateModel!(testmodel,i,X,y)
        final_t = time_ns()
        tmod[i] = (final_t-init_t)/1e9
    end # of the loop over the folds
    if iFold > 1
        tmod[1] = tmod[2]
    end
    f_t = mean(tmod)
    writedlm("results/time_correction/$(testmodel.DatasetName)$(testmodel.MethodName).txt",f_t)

end #Loop over the models
