# using PyPlot
using PyCall; unshift!(PyVector(pyimport("sys")["path"]), "/home/theo/PGGPC/Code/src/TTGPC");
@pyimport TTGP;
doPlot = false
function get_Dataset(datasetname::String)
    data = readdlm("../data/"*datasetname)
    X = data[:,1:end-1]; y = data[:,end];
    return (X,y,datasetname)
end
# logx = 0#Array{Float64,1}()
function logit(x)
    return 1./(1+exp.(-x))
end
(X_data,y_data,DatasetName) = get_Dataset("Diabetis")
# data = readdlm("test_file_circle")
# X_data = data[:,1:end-1]; y_data = data[:,end];
# DatasetName = "Test_Circle"
MaxIter = 10000 #Maximum number of iterations for every algorithm
(nSamples,nFeatures) = size(X_data);
nFold = 10; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
#Global variables for debugging
X = []; y = []; X_test = []; y_test = [];i=4
X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
M=10; θ=15; ϵ=1e-4; γ=1e-3
BatchSize = 100
InducingPoints = KMeansInducingPoints(X,M,10)
GPKernel = gpflow.kernels[:RBF](nFeatures,lengthscales=1.0*ones(nFeatures))#Take number of parameters as input
# GPKernel = gpflow.kernels[:Add]([gpflow.kernels[:RBF](nFeatures),gpflow.kernels[:White](input_dim=nFeatures,variance=γ)])#Take number of parameters as input
invKmm = Matrix(Symmetric(inv(GPKernel[:compute_K](InducingPoints,InducingPoints))))
K_starM = GPKernel[:compute_K](X_test,InducingPoints)
K_starstar = GPKernel[:compute_Kdiag](X_test,X_test)
 logt = Array{Float64,1}()
 logf = Array{Float64,1}()
 logdifft = Array{Float64,1}()
 logacc = Array{Float64,1}()
 logpred = Array{Any,1}()
 st = time_ns()
 function ComputeLogLikelihood(y_predic,y_test)
     y_new = zeros(y_predic)
     y_new[y_test.==1] = log.(y_predic[y_test.==1])
     y_new[y_test.==-1] = log.(1-y_predic[y_test.==-1])
     return y_new
end
 tic()
model = gpflow.models[:SVGP](X, reshape((y+1)./2,(length(y),1)), kern=GPKernel, likelihood=gpflow.likelihoods[:Bernoulli](), Z=InducingPoints, minibatch_size=BatchSize)
logconv= []
q_prev= Inf*ones(2*M)
@pydef type GPFlowToolSet <: gpflow.actions[:Action]
    __init__(self,model,text) = begin
        self[:model] = model
        self[:text] = text
        self[:i] = 1
    end
    run(self,ctx) =  begin
        # push!(logx,x)
        likelihood = ctx[:session][:run](self[:model][:likelihood_tensor])
        print("$(self[:text]) : likelihood $likelihood")
        if self[:i]%1 == 0
             push!(logt,(time_ns()-st))
             q_new = [model[:q_mu][:value][:,1];diag(model[:q_sqrt][:value][:,:,1])]
             conv = mean(abs.(q_new-q_prev)./((abs.(q_prev)+abs.(q_new))/2.0))
             q_prev[:] = q_new
        #     # println(q_prev)
             push!(logconv,q_new)
        #     println("$(self[:i]) : $conv")
        #     if conv < ϵ
        #         return false
        #     end
        #     # y_SSGPC,dummy = model[:predict_y](X_test) #return mean and variance
        #     # push!(logf,model[:_objective](x)[1])
        #     # y_new = ComputeLogLikelihood(y_SSGPC,y_test)
        #     # push!(logpred,median(y_new))
        end #endif
        self[:i]+=1
        return true
    end

end
function run_nat_grads_with_adam(m, lr, gamma, iterations; ind_points_fixed=true, kernel_fixed =false, var_list=nothing, callback=nothing)
    # we'll make use of this later when we use a XiTransform
    if var_list==nothing
        var_list = [(m[:q_mu], m[:q_sqrt])]
    end
    # we don't want adam optimizing these
    m[:q_mu][:set_trainable](false)
    m[:q_sqrt][:set_trainable](false)
    #
    ind_points_fixed ? model[:feature][:set_trainable](false) : nothing
    kernel_fixed ? model[:kern][:set_trainable](false) : nothing
    if ind_points_fixed || kernel_fixed
        adam = gpflow.train[:AdamOptimizer](lr)[:make_optimize_action](m)
    end
    natgrad = gpflow.training[:NatGradOptimizer](gamma=gamma)[:make_optimize_action](m, var_list=var_list)
    #
    # actions = [adam]
    if isdef(adam)
        actions = [adam, natgrad]
    else
        actions = [natgrad]
    end
    actions = callback!=nothing ? vcat(actions,callback) : actions
    gpflow.actions[:Loop](actions, stop=iterations)()
    model[:anchor](model[:enquire_session]())
end
a = GPFlowToolSet(model,"The ultimate test ")
# model[:optimize](callback=a[:getlog],method=tensorflow.train[:AdamOptimizer](),maxiter=MaxIter)
# gpflow.train[:AdamOptimizer](0.001)[:minimize](model,maxiter=100)
run_nat_grads_with_adam(model,0.001,1.,100,callback=a)
(y_SSGPC,) = model[:predict_y](X_test) #return mean and variance
y_SSGPC = sign.(y_SSGPC.*2-1)
acc= 1-sum(1-y_test.*y_SSGPC)/(2*length(y_test))
println("Acc is $acc")
figure(2);clf();
plot(logpred)
# plot(100,acc,marker="o",color="r")
# xlim([0,120])
# ylim([0,1])
toc()
