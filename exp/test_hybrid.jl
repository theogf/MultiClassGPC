using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using PyCall
using ValueHistories
using Plots
using LinearAlgebra
using BenchmarkTools
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
art_noise = 0.1

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
    global cols = reshape([RGB(permutedims(Vector(py_fgrid[i,:]))[collect(values(sort(model.ind_mapping)))]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
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

""

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
    AugmentedGaussianProcesses.computeMatrices!(model)
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
                push!(lparams,Symbol("l",i,"_",j),p_j)
            end
        else
            push!(lparams,Symbol("l",i),p[1])
        end
        push!(vparams,Symbol("v",i),getvariance(model.kernel[i]))
        push!(params,Symbol("Σ",i),tr(model.Σ[i]))
        push!(params,Symbol("μ",i),mean(model.μ[i]))
    end
    # println(model.kernel)
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

kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim,variance=10.0)
# kernel = AugmentedGaussianProcesses.RBFKernel(l,variance=10.0)
# setfixed!(kernel.fields.lengthscales)
# setfixed!(kernel.fields.variance)
autotuning = true

function trainhybrid(model,iterationsaug,iterations,callback,param)
    model.train(iterations=iterationsaug,callback=callback)
    new_model = AugmentedGaussianProcesses.SparseLogisticSoftMaxMultiClass(model.X,model.y,Stochastic=model.Stochastic,m=model.nFeatures,batchsize=model.nSamplesUsed,kernel=model.kernel[1],IndependentGPs=model.IndependentGPs,verbose=3,nEpochs=50,optimizer=0.5,Autotuning=true)
    for field in fieldnames(typeof(model))
        if field != :prev_params && field != :K_map && field != :g && field!= :h && field!= :τ && field != :Knn && field != :invK   && field != :train && field != :fstar && field != :predict && field != :predictproba && field != :elbo && field != :Autotuning
            @eval $new_model.$field = $model.$field
        end
    end
    println(model.kernel)
    model = new_model
    model.train(iterations=iterations,callback=callback)
    return model
end

### AUG. LOGISTIC SOFTMAX
elbos = MVHistory()
metrics = MVHistory()
lparams = MVHistory()
vparams = MVHistory()
params = MVHistory()

alsmmodel = AugmentedGaussianProcesses.SparseMultiClass(X,y,verbose=2,ϵ=1e-20,kernel=kernel,Autotuning=true,AutotuningFrequency=1,IndependentGPs=true,m=m)
Z = copy(alsmmodel.inducingPoints)
it_aug = 100
it = 200
model=alsmmodel
# alsmmodel.train(iterations=200,callback=callback)

alsmmodel = trainhybrid(alsmmodel,it_aug,it,callback,1)
# alsmmodel.train(iterations=10)
# @profiler alsmmodel.train(iterations=10)
# @profiler AugmentedGaussianProcesses.Gradient_Expec(alsmmodel)
# @btime AugmentedGaussianProcesses.Gradient_Expec(alsmmodel)
global py_alsm = alsmmodel.predictproba(X_test)
global y_alsm = alsmmodel.predict(X_test)
# println("Expected model Accuracy is $(acc(y_test,y_alsm)) and loglike : $(loglike(y_test,py_alsm)) in $t_alsm s")
##
alsm_map = title!(callbackplot(alsmmodel,2),"Hybrid LogSoftMax");
alsm_metrics = deepcopy(metrics)
alsm_lparams = deepcopy(lparams)
alsm_vparams = deepcopy(vparams)
alsm_params = deepcopy(params)
alsm_elbo = deepcopy(elbos)

pmet_alsm = plot(alsm_metrics,title="Hybrid LogSoftMax",markersize=0.0,linewidth=2.0)
vline!(pmet_alsm,[it_aug],lab="")

pl_alsm = plot(alsm_lparams,title="Hybrid LogSoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
vline!(pl_alsm,[it_aug],lab="")

pv_alsm = plot(alsm_vparams,title="Hybrid LogSoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
vline!(pv_alsm,[it_aug],lab="")

ppar_alsm = plot(alsm_params,title="Hybrid LogSoftMax",markersize=0.0,linewidth=2.0)
vline!(ppar_alsm,[it_aug],lab="")


pelbo_alsm = plot(alsm_elbo,title="Hybrid LogSoftMax",markersize=0.0,linewidth=2.0)
vline!(pelbo_alsm,[it_aug],lab="")
display(pmet_alsm)
display(pl_alsm)
display(pv_alsm)
display(ppar_alsm)
display(pelbo_alsm)
