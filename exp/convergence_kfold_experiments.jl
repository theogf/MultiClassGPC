#### Paper_Experiments ####
# Run on a file and compute Accuracy, loglikelihood and AUC on a nFold cross validation

include("get_arguments.jl")
cd(dirname(@__FILE__))
# if !in(LOAD_PATH,SRC_PATH); push!(LOAD_PATH,SRC_PATH); end;
#Compare SCGPMC, BSVM, SVGPMC and others

#Methods and scores to test
doSCGPMC = !args["SCGP"] #Sparse SCGPMC (sparsity)
doHSCGPMC = args["HSCGP"]
doEPGPMC = !args["EPGP"]
doSVGPMC = !args["SVGP"] #Sparse Variational GPMC (Hensmann)
doARMC = args["AR"]
doTTGPMC = args["TTGP"]

doBCGPMC = false
doStochastic = !args["stochastic"]
doAutotuning = !args["autotuning"]
doPointOptimization = args["point-optimization"]

include("functions_paper_experiment.jl")

ExperimentName = "Convergence"
doSaveLastState = args["last-state"]
doPlot = args["plot"]
doWrite = !args["no-writing"] #Write results in approprate folder
ShowIntResults = true #Show intermediate time, and results for each fold


#Testing Parameters
#= Datasets available are X :
aXa, Bank_marketing, Click_Prediction, Cod-rna, Diabetis, Electricity, German, Shuttle
=#
dataset = "segment"
# dataset = args["dataset"]
(X_data,y_data,DatasetName) = get_Dataset(dataset)
MaxIter = 500;#args["maxiter"] #Maximum number of iterations for every algorithm
iter_points= vcat(1:9,10:5:99,100:50:999,1e3:1e3:(1e4-1),1e4:1e4:1e5)

(nSamples,nFeatures) = size(X_data);
nFold = 10;#args["nFold"]; #Choose the number of folds
iFold = 10;#args["iFold"] > nFold ? nFold : args["iFold"]; #Number of fold to estimate
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
main_param["γ"] = 0.0
main_param["M"] = args["indpoints"]!=0 ? args["indpoints"] : min(100,floor(Int64,0.2*nSamples)) #Number of inducing points
main_param["Kernel"] = "ard"
l = initial_lengthscale(X_data)
main_param["Θ"] = sqrt(l) #initial Hyperparameter of the kernel
main_param["var"] = 1.0 #Variance
main_param["nClasses"] = length(unique(y_data))
main_param["BatchSize"] = 100;#args["batchsize"]
main_param["Verbose"] = 1
main_param["Window"] = 10
main_param["Autotuning"] = doAutotuning
main_param["PointOptimization"] = doPointOptimization
#All Parameters
BCGPMCParam = CGPMCParameters(main_param=main_param,independent=true)
SCGPMCParam = CGPMCParameters(Stochastic=doStochastic,Sparse=true,ALR=true,main_param=main_param,independent=true)
HSCGPMCParam = CGPMCParameters(dohybrid=true,Stochastic=doStochastic,Sparse=true,ALR=true,main_param=main_param,independent=true)
SVGPMCParam = SVGPMCParameters(Stochastic=doStochastic,main_param=main_param)
EPGPMCParam = EPGPMCParameters(Stochastic=doStochastic,main_param=main_param)


#Set of all models
TestModels = Dict{String,TestingModel}()

if doBCGPMC; TestModels["BCGPMC"] = TestingModel("BCGPMC",DatasetName,ExperimentName,"BCGPMC",BCGPMCParam); end;
if doSCGPMC; TestModels["SCGPMC"] = TestingModel("SCGPMC",DatasetName,ExperimentName,"SCGPMC",SCGPMCParam); end;
if doHSCGPMC; TestModels["HSCGPMC"] = TestingModel("HSCGPMC",DatasetName,ExperimentName,"HSCGPMC",HSCGPMCParam); end;
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
    testmodel.Results["Time"] = Vector{Vector{Float64}}();
    testmodel.Results["Accuracy"] = Vector{Vector{Float64}}();
    testmodel.Results["MeanL"] = Vector{Vector{Float64}}();
    testmodel.Results["MedianL"] = Vector{Vector{Float64}}();
    testmodel.Results["ELBO"] = Vector{Vector{Float64}}();
    testmodel.Results["AUC"] = Vector{Vector{Float64}}();
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
        try
            CreateModel!(testmodel,i,X,y)
            init_t = time_ns()
            if testmodel.MethodType == "EPGPMC"
                global LogArrays=copy(transpose(TrainModel!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)))
                push!(testmodel.Results["Time"],LogArrays[1,:])
                push!(testmodel.Results["AUC"],LogArrays[6,:])
            else
                global LogArrays= hcat(TrainModel!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)...)
                push!(testmodel.Results["Time"],TreatTime(init_t,LogArrays[1,:],LogArrays[6,:]))
                push!(testmodel.Results["AUC"],LogArrays[7,:])
            end
            push!(testmodel.Results["Accuracy"],LogArrays[2,:])
            push!(testmodel.Results["MeanL"],LogArrays[3,:])
            push!(testmodel.Results["MedianL"],LogArrays[4,:])
            push!(testmodel.Results["ELBO"],LogArrays[5,:])
            if ShowIntResults
                println("$(testmodel.MethodName) : Time  = $((time_ns()-init_t)*1e-9)s, accuracy : $(LogArrays[2,end])")
            end
        catch e
            # rethrow()
        end
        #Reset the kernel
        if testmodel.MethodName == "SVGPMC"
            testmodel.Param["Kernel"] = gpflow.kernels[:Sum]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=true),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
        elseif testmodel.MethodName == "SCGPMC"
            println("SCGPMC : params : $([AugmentedGaussianProcesses.KernelModule.getlengthscales(k) for k in testmodel.Model[i].kernel])\n and coeffs :  $([AugmentedGaussianProcesses.KernelModule.getvariance(k) for k in testmodel.Model[i].kernel])")
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
    using PyPlot
    PlotResults(TestModels)
end
