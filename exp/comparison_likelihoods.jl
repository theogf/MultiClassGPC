using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using PyCall
using ValueHistories
using Plots
using LinearAlgebra
include("metrics.jl")
pyplot()
clibrary(:cmocean)
seed!(42)
@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp
@pyimport gpflow
@pyimport tensorflow as tf
function run_nat_grads_with_adam(model,iterations; ind_points_fixed=true, kernel_fixed =false, callback=nothing , Stochastic = true)
    # we'll make use of this later when we use a XiTransform

    gamma_start = 1e-4;
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
            println("Gamma $g on iteration $i is too big: Falling back to $(g*gamma_fallback)")
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

N_data = 1000
N_class = 3
N_test = 50
N_grid = 100
minx=-5.0
maxx=5.0
noise = 1.0
truthknown = false
doMCCompare = false
dolikelihood = false

function latent(X)
    return sqrt.(X[:,1].^2+X[:,2].^2)
end
N_dim=2
N_iterations = 500
m = 50
art_noise = 0.5

X_clean = (rand(N_data,N_dim)*2.0).-1.0
y = zeros(Int64,N_data); y_noise = similar(y)
function classify(X,y)
    for i in 1:size(X,1)
        if X[i,2] < min(0,-X[i,1])
            y[i] = 1
        elseif X[i,2] > max(0,X[i,1])
            y[i] = 2
        else
            y[i] = 3
        end
    end
    return y
end
X= X_clean+rand(Normal(0,art_noise),N_data,N_dim)
classify(X_clean,y);classify(X,y_noise);
bayes_error = count(y.!=y_noise)/length(y)

xmin = minimum(X); xmax = maximum(X)
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
x_grid = range(xmin,length=N_grid,stop=xmax)
X_grid = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])

metrics = MVHistory()
kerparams = MVHistory()
elbos = MVHistory()
anim  = Animation()
function callbackplot(model,iter)
    if iter%2 !=0
        return
    end
    y_fgrid =  model.predict(X_grid)
    global py_fgrid = model.predictproba(X_grid)
    global cols = reshape([RGB(vec(convert(Array,py_fgrid[i,:]))[collect(values(sort(model.ind_mapping)))]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
    col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global p1= plot(x_grid,x_grid,cols,t=:contour,colorbar=false,framestyle=:box)
    lims = (xlims(p1),ylims(p1))
    p1=plot!(p1,X[:,1],X[:,2],color=col_doc[y],t=:scatter,lab="",markerstrokewidth=0.2)
    p1=plot!(p1,model.inducingPoints[1][:,1],model.inducingPoints[1][:,2],color=:black,t=:scatter,lab="")
    p1= plot!(x_grid,x_grid,reshape(y_fgrid,N_grid,N_grid),clims=(0,100),t=:contour,colorbar=false,color=:gray,levels=10)
    xlims!(p1,lims[1]);ylims!(p1,lims[2])
    frame(anim,p1)
    display(p1)
    return p1
end

##

function gpflowcallbackplot(model,iter)
    if iter%2 !=0
        return
    end
    global py_fgrid = rmmodel[:predict_y](X_grid)[1]
    y_fgrid = mapslices(argmax,py_fgrid,dims=2)
    global cols = reshape([RGB(py_fgrid[i,:]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
    col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global p1= plot(x_grid,x_grid,cols,t=:contour,colorbar=false,framestyle=:box)
    lims = (xlims(p1),ylims(p1))
    p1=plot!(p1,X[:,1],X[:,2],color=col_doc[y],t=:scatter,lab="",markerstrokewidth=0.2)
    p1=plot!(p1,model[:feature][:Z][:value][:,1],model[:feature][:Z][:value][:,2],color=:black,t=:scatter,lab="")
    p1= plot!(x_grid,x_grid,reshape(y_fgrid,N_grid,N_grid),clims=(-0,100),t=:contour,colorbar=false,color=:gray,levels=10)
    xlims!(p1,lims[1]);ylims!(p1,lims[2])
    frame(anim,p1)
    display(p1)
    return p1
end

function acc(y_test,y_pred)
    count(y_test.==y_pred)/length(y_pred)
end

function loglike(y_test,y_pred)
    ll = 0.0
    for i in 1:length(y_test)
        ll += log(y_pred[Symbol(y_test[i])][i])
    end
    ll /= length(y_test)
    return ll
end

function gpflowacc(y_test,y_pred)
    score = 0.0
    for i in 1:length(y_test)
        if argmax(y_pred[i,:])==y_test[i]
            score += 1
        end
    end
    return score/length(y_test)
end

function gpflowloglike(y_test,y_pred)
    score = 0.0
    for i in 1:length(y_test)
        score += log(y_pred[i,y_test[i]])
    end
    return score/length(y_test)
end

function callback(model,iter)
    y_pred = model.predict(X_test)
    py_pred = model.predictproba(X_test)
    push!(metrics,:err,1-acc(y_test,y_pred))
    push!(metrics,:ll,-loglike(y_test,py_pred))
    push!(elbos,:ELBO,-ELBO(model))
    push!(elbos,:NegGaussianKL,-AugmentedGaussianProcesses.GaussianKL(model))
    push!(elbos,:ExpecLogLike,AugmentedGaussianProcesses.ExpecLogLikelihood(model))
    for i in 1:model.K
        p = getlengthscales(model.kernel[i])
        if length(p) > 1
            for (j,p_j) in enumerate(p)
                push!(kerparams,Symbol("l",i,"_",j),p_j)
            end
        else
            push!(kerparams,Symbol("l",i),p[1])
        end
        push!(kerparams,Symbol("v",i),getvariance(model.kernel[i]))
    end
end

function callbackgpflow(model,session,iter)
      y_p = model[:predict_y](X_test)[1]
      push!(metrics,:ll,-gpflowloglike(y_test,y_p))
      push!(metrics,:err,1-gpflowacc(y_test,y_p))
      push!(elbos,:ELBO,session[:run](model[:likelihood_tensor]))
      model[:anchor](session)
      p = Array(model[:kern][:lengthscales][:value])
      for (j,p_j) in enumerate(p)
          push!(kerparams,Symbol("l_",j),p_j)
      end
      push!(kerparams,:v,Array(model[:kern][:variance][:value])[1])
end

function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X')
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
l = sqrt(initial_lengthscale(X))

kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim,variance=1.0)
# kernel = AugmentedGaussianProcesses.RBFKernel(l,variance=10.0)
# setfixed!(kernel.fields.lengthscales)
# setfixed!(kernel.fields.variance)
nBins = 15
autotuning = true


### AUG. LOGISTIC SOFTMAX
elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()

alsmmodel = AugmentedGaussianProcesses.SparseMultiClass(X,y,verbose=0,ϵ=1e-20,kernel=kernel,Autotuning=autotuning,AutotuningFrequency=1,IndependentGPs=true,m=m)
Z = copy(alsmmodel.inducingPoints)
alsmmodel.train(iterations=1)
# @profiler t_alsm = @elapsed alsmmodel.train(iterations=100)
# Atom.@trace AugmentedGaussianProcesses.updateHyperParameters!(alsmmodel)
t_alsm = @elapsed alsmmodel.train(iterations=N_iterations,callback=callback)

global py_alsm = alsmmodel.predictproba(X_test)
global y_alsm = alsmmodel.predict(X_test)
AUC_alsm = 0
println("Expected model accuracy is $(acc(y_test,y_alsm)), loglike : $(loglike(y_test,py_alsm)) and AUC $(AUC_alsm) in $t_alsm s")
alsm_map = title!(callbackplot(alsmmodel,2),"Aug. LogSoftMax")
alsm_metrics = deepcopy(metrics)
alsm_kerparams = deepcopy(kerparams)
alsm_elbo = deepcopy(elbos)
ECE_alsm, MCE_alsm, cal_alsm, calh_alsm =calibration(y_test,py_alsm,nBins=nBins,plothist=true,plotline=true)


## LOGISTIC SOFTMAX
elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()

lsmmodel = AugmentedGaussianProcesses.SparseLogisticSoftMaxMultiClass(X,y,verbose=2,ϵ=1e-20,kernel=kernel,optimizer=0.1,Autotuning=autotuning,AutotuningFrequency=1,IndependentGPs=true,m=m)
lsmmodel.inducingPoints = Z
t_lsm = @elapsed lsmmodel.train(iterations=N_iterations,callback=callback)

global py_lsm = lsmmodel.predictproba(X_test)
global y_lsm = lsmmodel.predict(X_test)
AUC_lsm = 0#multiclassAUC(lsmmodel,y_test,py_lsm)
println("Expected model accuracy is $(acc(y_test,y_lsm)), loglike : $(loglike(y_test,py_lsm)) and AUC $(AUC_lsm) in $t_lsm s")
lsm_map = title!(callbackplot(lsmmodel,2),"LogSoftMax")
lsm_metrics = deepcopy(metrics)
lsm_kerparams = deepcopy(kerparams)
lsm_elbo = deepcopy(elbos)
ECE_lsm, MCE_lsm, cal_lsm, calh_lsm = calibration(y_test,py_lsm,nBins=nBins,plothist=true,plotline=true)
## SOFTMAX
elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()
smmodel = AugmentedGaussianProcesses.SparseSoftMaxMultiClass(X,y,verbose=2,ϵ=1e-20,kernel=kernel,optimizer=0.1,Autotuning=autotuning,AutotuningFrequency=1,IndependentGPs=true,m=m)
smmodel.inducingPoints = Z
t_sm = @elapsed smmodel.train(iterations=N_iterations,callback=callback)

global py_sm = smmodel.predictproba(X_test)
global y_sm = smmodel.predict(X_test)
AUC_sm = 0;#multiclassAUC(smmodel,y_test,py_sm)
println("Expected model accuracy is $(acc(y_test,y_sm)), loglike : $(loglike(y_test,py_sm)) and AUC $(AUC_sm) in $t_sm s")
sm_map= title!(callbackplot(smmodel,2),"SoftMax")
sm_metrics = deepcopy(metrics)
sm_kerparams = deepcopy(kerparams)
sm_elbo = deepcopy(elbos)
ECE_sm, MCE_sm, cal_sm, calh_sm = calibration(y_test,py_sm,nBins=nBins,plothist=true,plotline=true)

## ROBUST MAX
elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()
rmmodel = gpflow.models[:SVGP](X, Float64.(reshape(y.-1,(length(y),1))),kern=gpflow.kernels[:RBF](N_dim,lengthscales=l,ARD=true),likelihood=gpflow.likelihoods[:MultiClass](N_class),num_latent=N_class,Z=Z[1])
t_rm = @elapsed run_nat_grads_with_adam(rmmodel,N_iterations/10,callback=callbackgpflow,Stochastic=false)

global py_rm = rmmodel[:predict_y](X_test)[1]
AUC_rm = 0#multiclassAUC(y_test,py_rm)
println("Expected model accuracy is $(gpflowacc(y_test,py_rm)), loglike : $(gpflowloglike(y_test,py_rm)) and AUC $(AUC_rm) in $t_sm s")
rm_map= title!(gpflowcallbackplot(rmmodel,2),"RobustMax")
rm_metrics = deepcopy(metrics)
rm_kerparams = deepcopy(kerparams)
rm_elbo = deepcopy(elbos)
ECE_rm, MCE_rm, cal_rm, calh_rm = calibration(y_test,py_rm,nBins=nBins,plothist=true,plotline=true,gpflow=true)


## Plotting part
pmet_alsm = plot(alsm_metrics,title="Aug. LogSoftMax",markersize=0.0,linewidth=2.0)
pmet_lsm = hline!(pmet_alsm,[bayes_error],lab="",line=(2.0,:red))
pmet_lsm = plot(lsm_metrics,title="LogSoftMax",markersize=0.0,linewidth=2.0)
pmet_lsm = hline!(pmet_lsm,[bayes_error],lab="",line=(2.0,:red))
pmet_sm = plot(sm_metrics,title="SoftMax",markersize=0.0,linewidth=2.0)
pmet_sm = hline!(pmet_sm,[bayes_error],lab="",line=(2.0,:red))
pmet_rm = plot(rm_metrics,title="RobustMax",markersize=0.0,linewidth=2.0)
pmet_rm = hline!(pmet_rm,[bayes_error],lab="",line=(2.0,:red))
met_lims = (min(ylims(pmet_alsm)[1],ylims(pmet_lsm)[1],ylims(pmet_sm)[1],ylims(pmet_rm)[1]),max(ylims(pmet_alsm)[2],ylims(pmet_lsm)[2],ylims(pmet_sm)[2],ylims(pmet_rm)[2]))

pker_alsm = plot(alsm_kerparams,title="Aug. LogSoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
pker_lsm = plot(lsm_kerparams,title="LogSoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
pker_sm = plot(sm_kerparams,title="SoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
pker_rm = plot(rm_kerparams,title="RobustMax",yaxis=:log,markersize=0.0,linewidth=2.0)
ker_lims = (min(ylims(pker_alsm)[1],ylims(pker_lsm)[1],ylims(pker_sm)[1],ylims(pker_rm)[1]),max(ylims(pker_alsm)[2],ylims(pker_lsm)[2],ylims(pker_sm)[2],ylims(pker_rm)[2]))

pelbo_alsm = plot(lsm_elbo,title="Aug. LogSoftMax",markersize=0.0,linewidth=2.0)
pelbo_lsm = plot(lsm_elbo,title="LogSoftMax",markersize=0.0,linewidth=2.0)
pelbo_sm = plot(sm_elbo,title="SoftMax",markersize=0.0,linewidth=2.0)
pelbo_rm = plot(rm_elbo,title="RobustMax",markersize=0.0,linewidth=2.0)
elbo_lims = (min(ylims(pelbo_alsm)[1],ylims(pelbo_lsm)[1],ylims(pelbo_sm)[1],ylims(pelbo_rm)[1]),max(ylims(pelbo_alsm)[2],ylims(pelbo_lsm)[2],ylims(pelbo_sm)[2],ylims(pelbo_rm)[2]))

pmet = plot(ylims!(pmet_alsm,met_lims),ylims!(pmet_lsm,met_lims),ylims!(pmet_sm,met_lims),ylims!(pmet_rm,met_lims))
pker = plot(ylims!(pker_alsm,ker_lims),ylims!(pker_lsm,ker_lims),ylims!(pker_sm,ker_lims),ylims!(pker_rm,ker_lims))
pelbo = plot(ylims!(pelbo_alsm,elbo_lims),ylims!(pelbo_lsm,elbo_lims),ylims!(pelbo_sm,elbo_lims),ylims!(pelbo_rm,elbo_lims))
pmap = plot(alsm_map,lsm_map,sm_map,rm_map)

methods_name = ["Aug. LogSoftMax","LogSoftMax","SoftMax","RobustMax"]
pAUC = bar(methods_name,[AUC_alsm,AUC_lsm,AUC_sm,AUC_rm],lab="",title="MultiClass AUC")
pll = bar(methods_name,[alsm_metrics[:ll].values[end],lsm_metrics[:ll].values[end],sm_metrics[:ll].values[end],rm_metrics[:ll].values[end]],lab="",title="Negative Log Likelihood")
perr = bar(methods_name,[alsm_metrics[:err].values[end],lsm_metrics[:err].values[end],sm_metrics[:err].values[end],rm_metrics[:err].values[end]],lab="",title="Error rate")
pmetfin = plot(perr,pll,pAUC)

display(pmet)
display(pmetfin)
display(pker)
display(pelbo)
display(pmap)
plot(pmet,pelbo)

cd(@__DIR__)
savefig(pmet,"resultslikelihood/metrics_noise$(art_noise).png")
savefig(pmetfin,"resultslikelihood/metricsfinal_noise$(art_noise).png")
savefig(pmap,"resultslikelihood/plot_noise$(art_noise).png")
savefig(pelbo,"resultslikelihood/elbo_noise$(art_noise).png")
savefig(pker,"resultslikelihood/kernel_params_noise$(art_noise).png")
