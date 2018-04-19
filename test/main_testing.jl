# if !isdefined(:DataAccess); include("DataAccess.jl"); end;
# if !isdefined(:PolyaGammaGPC); include("../src/XGPC.jl"); end;
# if !isdefined(:KernelFunctions); include("KernelFunctions.jl"); end;
 # include("../src/XGPC.jl");
# include("../src/DataAugmentedClassifiers.jl")
# include("../src/DataAugmentedClassifierFunctions.jl")
push!(LOAD_PATH,".")
using PyPlot
using DataAccess
using KernelFunctions
using DataAugmentedClassifiers
doPlot = false
(X_data,y_data,DatasetName) = get_Dataset("aXa")
# data = readdlm("test_file_circle")
# X_data = data[:,1:end-1]; y_data = data[:,end];
# DatasetName = "Test_Circle"
MaxIter = 10000#Maximum number of iterations for every algorithm
(nSamples,nFeatures) = size(X_data);
nFold = 10; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
#Global variables for debugging
X = []; y = []; X_test = []; y_test = [];i=4
X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
M=100; θ=5; ϵ=1e-10; γ=1e-3
kerns = [Kernel("rbf",1.0;params=θ)]
# kerns = [Kernel("linear",1.0)]
BatchSize = 100
Ninducingpoints = 100
# tic()
#  model_orig = XGPCOriginal(;Stochastic=false,batchSize=10,Sparse=false,m=M,
#  kernels=kerns,Autotuning=false,autotuningfrequency=10,AdaptativeLearningRate=false,κ_s=1.0,τ_s = 1,ϵ=ϵ,γ=γ,
#  κ_Θ=1.0,τ_Θ=1,smoothingWindow=10,VerboseLevel=0,Storing=true,StoringFrequency=1)
#  TrainXGPC(model_orig,X,y)
#  y_predic_orig = sign.(model_orig.Predict(X,X_test))
# # y_predic_prob = model_orig.PredictProba(X,X_test)
#  println(1-sum(1-y_test.*y_predic_orig)/(2*length(y_test)))
 # toc()
 tic()
 # model = BatchXGPC(X,y;Kernels=kerns,Autotuning=false,AutotuningFrequency=4,ρ_AT=0.05,VerboseLevel=3,ϵ=1e-3)
model = SparseXGPC(X,y;Stochastic=true,ϵ=1e-4,nEpochs=MaxIter,SmoothingWindow=10,Kernels=kerns,Autotuning=true,AutotuningFrequency=20,ρ_AT=0.01,VerboseLevel=2,AdaptiveLearningRate=true,BatchSize=BatchSize,m=Ninducingpoints,Storing=false)
LogArrays = Array{Any,1}()
LogQs = Array{Any,1}()
iter_points = 1:5:10000
function StoreIt(model::AugmentedClassifier,iter)#;iter_points=[],LogArrays=[],X_test=0,y_test=0)
    if in(iter,iter_points)
        a = zeros(6)
        a[1] = time_ns()
        y_p = model.PredictProba(X_test)
        loglike = zeros(y_p)
        loglike[y_test.==1] = log.(y_p[y_test.==1])
        loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
        a[2] = 1-sum(1-y_test.*sign.(y_p-0.5))/(2*length(y_test))
        a[3] = mean(loglike)
        a[4] = median(loglike)
        a[5] = ELBO(model)
        a[6] = time_ns()
        println("Iteration $iter : Accuracy is $(a[2]), ELBO is $(a[5]), θ is $(model.Kernels[1].param)")
        push!(LogArrays,a)
        push!(LogQs,[model.μ;diag(model.ζ)])
    end
end

function LogLikeConvergence(model::AugmentedClassifier,iter::Integer,X_test,y_test)
    if iter==1
        push!(model.evol_conv,Inf)
        y_p = model.PredictProba(X_test)
        loglike = zeros(y_p)
        loglike[y_test.==1] = log.(y_p[y_test.==1])
        loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
        new_params = mean(loglike)
        model.prev_params = new_params
        return Inf
    end
    if !model.Stochastic || iter%10 == 0
        y_p = model.PredictProba(X_test)
        loglike = zeros(y_p)
        loglike[y_test.==1] = log.(y_p[y_test.==1])
        loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
        new_params = mean(loglike)
        push!(model.evol_conv,abs(new_params-model.prev_params)/((abs(model.prev_params)+abs(new_params))/2.0))
        println("Last conv : $(model.evol_conv[end])")

        model.prev_params = new_params
    elseif model.Stochastic
        return 1
    end
    if model.Stochastic
        println("Averaged conv : $(mean(model.evol_conv[max(1,length(model.evol_conv)-model.SmoothingWindow+1):end]))")
        println("Windows goes from $(max(1,length(model.evol_conv)-model.SmoothingWindow+1)) to $(length(model.evol_conv))")
        return mean(model.evol_conv[max(1,length(model.evol_conv)-model.SmoothingWindow+1):end])
    else
        return model.evol_conv[end]
    end
end
# model = SparseBSVM(X,y;Stochastic=true,Kernels=kerns,Autotuning=true,SmoothingWindow=50,AutotuningFrequency=1,VerboseLevel=3,ρ_AT=0.2,AdaptiveLearningRate=true,BatchSize=50,m=50)
# model = BatchBSVM(X,y;Kernels=kerns,Autotuning=false,AutotuningFrequency=2,VerboseLevel=1)
 # model = LinearBSVM(X,y;Intercept=true,Stochastic=false,BatchSize=30,AdaptiveLearningRate=true,VerboseLevel=3,Autotuning=true,AutotuningFrequency=5)

 model.train(callback=StoreIt)
 # model.train(callback=StoreIt,convergence=function (model::AugmentedClassifier,iter)  return LogLikeConvergence(model,iter,X_test,y_test);end)
 y_predic_log = model.Predict(X_test)
println(1-sum(1-y_test.*y_predic_log)/(2*length(y_test)))
toc()
LogArrays = hcat(LogArrays...)
figure(2); clf();subplot(1,2,1); plot(LogArrays[3,:]); ylabel("Mean(Log likelihood)")
 subplot(1,2,2); plot(LogArrays[4,:]); ylabel("Median(Log likelihood)")
if doPlot
    figure(2);clf();
    #GIG
    b=model.α; a=1
    mean_GIG = sqrt.(b).*besselk.(1.5,sqrt.(a.*b))./(sqrt.(a).*besselk.(0.5,sqrt.(a.*b)))
    #PG
    mean_PG= 1.0./(2*model.α).*tanh.(model.α/2)
    scatter(X[:,1],X[:,2],c=mean_PG)
    circle = (0:360)/180*pi
    radius = 1.5
    # plot(radius*cos.(circle),radius*sin.(circle),color="k",marker="None",linewidth=1)
    # model.Plotting("logELBO")
    colorbar()
    xlim([-3,3])
    ylim([-3,3])
    figure(3);clf();
    evol_conv = zeros(MaxIter)
    for i in 1:MaxIter
        evol_conv[i] = Convergence(model,i)
    end
    semilogy(evol_conv)
end
