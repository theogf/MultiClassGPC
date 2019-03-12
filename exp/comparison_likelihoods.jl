using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using PyCall
using RCall
using ValueHistories
using Plots
using Makie
using LinearAlgebra
using GradDescent
using DelimitedFiles
cd(@__DIR__)
include("metrics.jl")
pyplot()
clibrary(:cmocean)
seed!(42)
@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp
@pyimport gpflow
@pyimport tensorflow as tf
R"source('../src/sepMGPC_batch.R')"
function run_nat_grads_with_adam(model,iterations; ind_points_fixed=true, kernel_fixed =false, callback=nothing , Stochastic = true)
    # we'll make use of this later when we use a XiTransform

    gamma_start = 1e-4;
    if Stochastic
        gamma_max = 1e-1;    gamma_step = 10^(0.1); gamma_fallback = 1e-2;
    else
        gamma_max = 0.1;    gamma_step = 10^(0.1); gamma_fallback = 1e-2;
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
        op_adam = gpflow.train[:AdamOptimizer](0.01)[:make_optimize_tensor](model)
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
            println("γ $g on iteration $i is too big: Falling back to $(g*gamma_fallback)")
            sess[:run](op_gamma_fallback)
          end
        end
        if op_adam!=0
            try
                sess[:run](op_adam)
            catch e
                if isa(a,InterruptException)
                    break;
                else
                    rethrow(e)
                end
            end
        end
        if i % 100 == 0
            println("$i γ=$(sess[:run](gamma)) ELBO=$(sess[:run](model[:likelihood_tensor]))")
        end
        if callback!= nothing
            if i%max(1,div(10^(floor(Int,log10(i))),2))==0
                callback(model,sess,i)
            end
        end
    end
    model[:anchor](sess)
end

N_data = 500
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
art_noise = 0.3
dpi=600
##
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
##
    σ = 0.5; N_class = N_dim+1
    centers = zeros(N_class,N_dim)
    for i in 1:N_dim
        centers[i,i] = 1
    end
    centers[end,:] = (1+sqrt(N_class))/N_dim*ones(N_dim)
    centers./= sqrt(N_dim)
    distr = [MvNormal(centers[i,:],σ) for i in 1:N_class]
    X = zeros(Float64,N_data,N_dim)
    y = zeros(Int64,N_data)
    true_py = zeros(Float64,N_data)
    for i in 1:N_data
        y[i] = rand(1:N_class)
        X[i,:] = rand(distr[y[i]])
        true_py[i] = pdf(distr[y[i]],X[i,:])/sum(pdf(distr[k],X[i,:]) for k in 1:N_class)
    end
    function plot_data(X,y,centers,distr)

    end
plot_data(X,y,centers,distr)

##
xmin = minimum(X); xmax = maximum(X)
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
x_grid = range(xmin,length=N_grid,stop=xmax)
X_grid = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])

metrics = MVHistory()
kerparams = MVHistory()
elbos = MVHistory()
anim  = Animation()
function callbackplot(model,iter,title)
    y_fgrid = predict_y(model,X_grid)
    global py_fgrid = proba_y(model,X_grid)
    global cols = reshape([RGB(permutedims(Vector(py_fgrid[i,:]))[collect(values(sort(model.likelihood.ind_mapping)))]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
    col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global p1= Plots.plot(x_grid,x_grid,cols,t=:contour,colorbar=false,grid=:hide,framestyle=:none,yflip=false,dpi=dpi,title=title,titlefontsize=tfontsize)
    lims = (xlims(p1),ylims(p1))
    p1=Plots.plot!(p1,X[:,1],X[:,2],color=col_doc[y],t=:scatter,lab="",markerstrokewidth=0.3)
    # p1=plot!(p1,model.Z[1][:,1],model.Z[1][:,2],color=:black,t=:scatter,lab="")
    p1= Plots.plot!(x_grid,x_grid,reshape(y_fgrid,N_grid,N_grid),clims=(0,100),t=:contour,colorbar=false,color=:gray,levels=10)
    xlims!(p1,lims[1]);ylims!(p1,lims[2])
    frame(anim,p1)
    return p1
end

function convert_liketoRGB(mu,py,colors)
    [py_fgrd for i in 1:N_grid*N_grid]
end

function callbackmakie(model)
    global y_fgrid = predict_y(model,X_grid)
    global py_fgrid = Matrix(proba_y(model,X_grid))[:,collect(values(sort(model.likelihood.ind_mapping)))]
    global μ_fgrid = predict_f(model,X_grid,covf=false)
    global cols = reshape([parse.(Colorant,RGB(py_fgrid[i,:]...)) for i in 1:N_grid*N_grid],N_grid,N_grid)
    global col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global scale = 1.0
    global scene = Scene()
    Makie.scatter!(scene,[1,0,0],[0,1,0],[0,0,1],color=RGBA(1,1,1,0)) #For 3D plots
    Makie.scatter!(scene,X[:,1],X[:,2],scale*(model.nLatent+1)*ones(size(X,1)),color=col_doc[y],lab="",markerstrokewidth=0.1,transparency=true,shading=false)
    Makie.surface!(scene,collect(x_grid),collect(x_grid),zeros(N_grid,N_grid),grid=:hide,color=cols',lab="",shading=false)#,size=(600,2000))#,colorbar=false,framestyle=:none,,dpi=dpi)
    Makie.lines!(scene,[xmin,xmin,xmax,xmax,xmin],[xmin,xmax,xmax,xmin,xmin],zeros(5),lab="",color=:black,linewidth=2.0,shading=false)
    tsize = 0.8
    minalpha = 0.2
    grads = [cgrad([RGBA(1,1,1,minalpha),RGBA(1,0,0,1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(0,1,0,1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(0,0,1,1)])]
    Makie.text!(scene,"p(y|D)",position=(xmin,xmax,0.0),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    sub = ["₃","₂","₁"]
    for i in 1:model.nLatent
        μ = μ_fgrid[collect(values(sort(model.likelihood.ind_mapping)))][i]
        μ = (μ.-minimum(μ))/(maximum(μ)-minimum(μ))
        int_cols = getindex.([grads[i]],μ)
        Makie.surface!(scene,collect(x_grid),collect(x_grid),scale*i*ones(N_grid,N_grid),color=reshape(int_cols,N_grid,N_grid)',shading=false)
        Makie.lines!(scene,[xmin,xmin,xmax,xmax,xmin],[xmin,xmax,xmax,xmin,xmin],scale*i*ones(5),lab="",color=:black,linewidth=2.0)
        Makie.text!(scene,"p(f"*sub[i]*"|D)",position = (xmin,xmax,scale*i),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    end

    Makie.lines!(scene,[xmin,xmin,xmax,xmax,xmin],[xmin,xmax,xmax,xmin,xmin],scale*(model.nLatent+1)*ones(5),lab="",color=:black,linewidth=2.0)
    scene[Axis][:showgrid] = (false,false,false)
    scene[Axis][:showaxis] = (false,false,false)
    scene[Axis][:ticks][:textsize] = 0
    scene[Axis][:names][:axisnames] = ("","","")
    Makie.text!(scene,"data",position = (xmin,xmax,scale*(model.nLatent+1)),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    scene.center=false
    return scene
end

function callbackplot3D(model,iter)
    global y_fgrid = predict_y(model,X_grid)
    global py_fgrid = proba_y(model,X_grid)
    global μ_fgrid = predict_f(model,X_grid,covf=false)
    global cols = reshape([RGB(permutedims(Vector(py_fgrid[i,:]))[collect(values(sort(model.likelihood.ind_mapping)))]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
    global col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global grad = cgrad(vec(cols))
    global scale = 4.0
    # global p3d= Plots.surface(collect(x_grid),collect(x_grid),scale*(model.nLatent+1)*ones(N_grid,N_grid),grid=:hide,color=grad)#,colorbar=false,framestyle=:none,yflip=false,dpi=dpi)
    global p3d= Plots.scatter3d(X_grid[:,1],X_grid[:,2],zeros(N_grid*N_grid),grid=:hide,color=grad.colors,marker=:square,markerstrokewidth=0,lab="")#,size=(600,2000))#,colorbar=false,framestyle=:none,,dpi=dpi)
    grads = [cgrad([RGBA(1,1,1,0.5),RGBA(1,0,0,1)]),cgrad([RGBA(1,1,1,0.5),RGBA(0,1,0,1)]),cgrad([RGBA(1,1,1,0.5),RGBA(0,0,1,1)])]
    for i in 1:model.nLatent
        Plots.surface!(p3d,collect(x_grid),collect(x_grid),scale*i*ones(N_grid,N_grid),colorbar=false,fill_z=reshape(μ_fgrid[collect(values(sort(model.likelihood.ind_mapping)))][i],N_grid,N_grid),color=grads[i],yflip=false)
    end
    # # lims = (xlims(p3d),ylims(p3d))
    global p3d = Plots.plot3d!([xmin,xmin,xmax,xmax,xmin],[xmin,xmax,xmax,xmin,xmin],scale*(model.nLatent+2)*ones(5),lab="",color=:black,linewidth=2.0)
    global p3d=Plots.scatter3d!(p3d,X[:,1],X[:,2],scale*(model.nLatent+2)*ones(size(X,1)),color=col_doc[y],lab="",markerstrokewidth=0.3,grid=:off,axis=:off)
    # xlims!(p3d,lims[1]);ylims!(p3d,lims[2])
    # pfinal= contour3d!(pfinal,x_grid,x_grid,(model.nLatent+1)*ones(length(x_grid)),reshape(y_fgrid,N_grid,N_grid),clims=(0,100),t=:contour,colorbar=false,color=:gray,levels=10)
    # p1=plot!(p1,model.Z[1][:,1],model.Z[1][:,2],color=:black,t=:scatter,lab="")
    # frame(anim,p3d)
    return p3d
end
##

function gpflowcallbackplot(model,iter,title)
    global py_fgrid = rmmodel[:predict_y](X_grid)[1]
    y_fgrid = mapslices(argmax,py_fgrid,dims=2)
    global cols = reshape([RGB(py_fgrid[i,:]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
    col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global p1= Plots.plot(x_grid,x_grid,cols,t=:contour,colorbar=false,grid=:hide,framestyle=:none,yflip=false,dpi=dpi,title=title,titlefontsize=tfontsize)
    lims = (xlims(p1),ylims(p1))
    p1=Plots.plot!(p1,X[:,1],X[:,2],color=col_doc[y],t=:scatter,lab="",markerstrokewidth=0.3)
    # p1=plot!(p1,model[:feature][:Z][:value][:,1],model[:feature][:Z][:value][:,2],color=:black,t=:scatter,lab="")
    p1= Plots.plot!(x_grid,x_grid,reshape(y_fgrid,N_grid,N_grid),clims=(-0,100),t=:contour,colorbar=false,color=:gray,levels=10)
    xlims!(p1,lims[1]);ylims!(p1,lims[2])
    frame(anim,p1)
    display(p1)
    return p1
end

function epcallbackplot(model,iter,title)
    global py_fgrid = Matrix(rcopy(R"predictMGPC($(model),$(X_grid))$prob"))
    y_fgrid = mapslices(argmax,py_fgrid,dims=2)
    global cols = reshape([RGB(py_fgrid[i,:]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
    col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global p1= plot(x_grid,x_grid,cols,t=:contour,colorbar=false,grid=:hide,framestyle=:none,yflip=false,dpi=dpi,title=title,titlefontsize=tfontsize)
    lims = (xlims(p1),ylims(p1))
    p1=plot!(p1,X[:,1],X[:,2],color=col_doc[y],t=:scatter,lab="",markerstrokewidth=0.3)
    # p1=plot!(p1,model[:feature][:Z][:value][:,1],model[:feature][:Z][:value][:,2],color=:black,t=:scatter,lab="")
    p1= plot!(x_grid,x_grid,reshape(y_fgrid,N_grid,N_grid),clims=(-0,100),t=:contour,colorbar=false,color=:gray,levels=10)
    xlims!(p1,lims[1]);ylims!(p1,lims[2])
    # frame(anim,p1)
    display(p1)
    return p1
end

function saveproba(y,y_test,σ,name,sc)
    if sc
        reorder = sortperm(parse.(Int64,string.(names(y))))
        y = Matrix(y)[:,reorder]
    end
    writedlm("resultslikelihood/y_proba_$(σ)_$(name).txt",hcat(y_test,y))
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
    if iter == 0 || iter%max(1,div(10^(floor(Int,log10(iter))),2))==0
        y_pred = predict_y(alsmmodel,X_test)
        py_pred = proba_y(alsmmodel,X_test)
        push!(metrics,:err,1-acc(y_test,y_pred))
        push!(metrics,:ll,-loglike(y_test,py_pred))
        push!(elbos,:ELBO,ELBO(model))
        push!(elbos,:NegGaussianKL,-AugmentedGaussianProcesses.GaussianKL(model))
        push!(elbos,:ExpecLogLike,AugmentedGaussianProcesses.expecLogLikelihood(model))
        for i in 1:model.nPrior
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
end

function callbackgpflow(model,session,iter)
      y_p = model[:predict_y](X_test)[1]
      push!(metrics,:ll,-gpflowloglike(y_test,y_p))
      push!(metrics,:err,1-gpflowacc(y_test,y_p))
      push!(elbos,:ELBO,session[:run](model[:likelihood_tensor]))
      model[:anchor](session)
      # p = Array(model[:kern][:lengthscales][:value])
      # for (j,p_j) in enumerate(p)
          # push!(kerparams,Symbol("l1_",j),p_j)
      # end
      # push!(kerparams,:v,Array(model[:kern][:variance][:value])[1])
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
nBins = 10
autotuning = true


## AUG. LOGISTIC SOFTMAX
elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()

alsmmodel = SVGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),AnalyticInference(),m,verbose=2,Autotuning=autotuning,IndependentPriors=!true)
Z = copy(alsmmodel.Z)
t_alsm = @elapsed train!(alsmmodel,iterations=N_iterations,callback=callback)

global py_alsm = proba_y(alsmmodel,X_test)
saveproba(py_alsm,y_test,σ,"alsm",true)
global y_alsm = predict_y(alsmmodel,X_test)
AUC_alsm = 0
println("Augmented model accuracy is $(acc(y_test,y_alsm)), loglike : $(loglike(y_test,py_alsm)) and AUC $(AUC_alsm) in $t_alsm s")
alsm_map = callbackplot(alsmmodel,2)
Plots.savefig(alsm_map,"../plotslikelihood/contour_alsm_σ$σ.pdf")
alsm_map = title!(alsm_map,"Aug. LogSoftMax")
alsm_metrics = deepcopy(metrics)
alsm_kerparams = deepcopy(kerparams)
alsm_elbo = deepcopy(elbos)
ECE_alsm, MCE_alsm, cal_alsm, calh_alsm =calibration(y_test,py_alsm,nBins=nBins,plothist=true,plotline=true,meanonly=true,threshold=2)
Plots.savefig(cal_alsm,"../plotslikelihood/cal_line_alsm_σ$σ.pdf")
Plots.savefig(calh_alsm,"../plotslikelihood/cal_hist_alsm_σ$σ.pdf")
## LOGISTIC SOFTMAX
elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()

lsmmodel = SVGP(X,y,kernel,LogisticSoftMaxLikelihood(),NumericalInference(:mcmc,nMC=1000,optimizer=VanillaGradDescent(η=0.01)),m,verbose=2,Autotuning=autotuning,IndependentPriors=!true)
lsmmodel.Z = Z
t_lsm = @elapsed train!(lsmmodel,iterations=N_iterations,callback=callback)
# @profiler train!(lsmmodel,iterations=2)
tfontsize=23
global py_lsm = proba_y(lsmmodel,X_test)
saveproba(py_lsm,y_test,σ,"lsm",true)
global y_lsm = predict_y(lsmmodel,X_test)
AUC_lsm = 0#multiclassAUC(lsmmodel,y_test,py_lsm)
println("Expected model accuracy is $(acc(y_test,y_lsm)), loglike : $(loglike(y_test,py_lsm)) and AUC $(AUC_lsm) in $t_lsm s")
lsm_map = callbackplot(lsmmodel,1,"Logistic-Softmax")
Plots.savefig(lsm_map,"../plotslikelihood/contour_lsm_σ$σ.pdf")
lsm_metrics = deepcopy(metrics)
lsm_kerparams = deepcopy(kerparams)
lsm_elbo = deepcopy(elbos)
# ECE_lsm, MCE_lsm, cal_lsm, calh_lsm, calconf_lsm = calibration(y_test,py_lsm,nBins=nBins,plothist=true,plotconf=true,plotline=true,meanonly=true,threshold=2)
# Plots.savefig(cal_lsm,"../plotslikelihood/cal_line_lsm_σ$σ.pdf")
# Plots.savefig(calh_lsm,"../plotslikelihood/cal_hist_lsm_σ$σ.pdf")
# Plots.savefig(calconf_lsm,"../plotslikelihood/cal_conf_lsm_σ$σ.pdf")

## SOFTMAX
elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()
smmodel = SVGP(X,y,kernel,SoftMaxLikelihood(),NumericalInference(:mcmc,optimizer=VanillaGradDescent(η=0.01)),m,verbose=2,Autotuning=autotuning,IndependentPriors=!true)
smmodel.Z = Z
t_sm = @elapsed train!(smmodel,iterations=N_iterations,callback=callback)
# @profiler train!(smmodel,iterations=1)
global py_sm = proba_y(smmodel,X_test)
saveproba(py_sm,y_test,σ,"sm",true)
global y_sm = predict_y(smmodel,X_test)
AUC_sm = 0;#multiclassAUC(smmodel,y_test,py_sm)
println("Expected model accuracy is $(acc(y_test,y_sm)), loglike : $(loglike(y_test,py_sm)) and AUC $(AUC_sm) in $t_sm s")
sm_map = callbackplot(smmodel,2,"Softmax")
Plots.savefig(sm_map,"../plotslikelihood/contour_sm_σ$σ.pdf")
sm_metrics = deepcopy(metrics)
sm_kerparams = deepcopy(kerparams)
sm_elbo = deepcopy(elbos)
# ECE_sm, MCE_sm, cal_sm, calh_sm, calconf_sm = calibration(y_test,py_sm,nBins=nBins,plothist=true,plotline=true,plotconf=true,meanonly=true,threshold=2)
# Plots.savefig(cal_sm,"../plotslikelihood/cal_line_sm_σ$σ.pdf")
# Plots.savefig(calh_sm,"../plotslikelihood/cal_hist_sm_σ$σ.pdf")
# Plots.savefig(calconf_sm,"../plotslikelihood/cal_conf_sm_σ$σ.pdf")
## ROBUST MAX
elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()
rmmodel = gpflow.models[:SVGP](X, Float64.(reshape(y.-1,(length(y),1))),kern=gpflow.kernels[:Sum]([gpflow.kernels[:RBF](N_dim,lengthscales=l,ARD=true),gpflow.kernels[:White](N_dim)]),likelihood=gpflow.likelihoods[:MultiClass](N_class),num_latent=N_class,Z=Z[1])
t_rm = @elapsed run_nat_grads_with_adam(rmmodel,N_iterations*10,callback=callbackgpflow,Stochastic=false)

global py_rm = rmmodel[:predict_y](X_test)[1]
saveproba(py_rm,y_test,σ,"rm",false)
AUC_rm = 0#multiclassAUC(y_test,py_rm)
println("Expected model accuracy is $(gpflowacc(y_test,py_rm)), loglike : $(gpflowloglike(y_test,py_rm)) and AUC $(AUC_rm) in $t_sm s")
rm_map = gpflowcallbackplot(rmmodel,1,"Robust-Max")
Plots.savefig(rm_map,"../plotslikelihood/contour_rm_σ$σ.pdf")
rm_metrics = deepcopy(metrics)
rm_kerparams = deepcopy(kerparams)
rm_elbo = deepcopy(elbos)
# ECE_rm, MCE_rm, cal_rm, calh_rm,calconf_rm = calibration(y_test,py_rm,nBins=nBins,plothist=true,plotline=true,plotconf=true,gpflow=true,meanonly=true,threshold=2)
# Plots.savefig(cal_rm,"../plotslikelihood/cal_line_rm_σ$σ.pdf")
# Plots.savefig(calh_rm,"../plotslikelihood/cal_hist_rm_σ$σ.pdf")
# Plots.savefig(calh_ep,"../plotslikelihood/cal_conf_rm_σ$σ.pdf")

## Multiclass-probit
elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()
#Xbar_ini=$(Z[1]),
t_ep = @elapsed epmodel = R"epMGPCInternal($X, $(y),$(size(Z[1],1)),  X_test = $X_test, Y_test= $(y_test),  max_iters=2000, indpoints= FALSE, autotuning=TRUE)"

global py_ep = Matrix(rcopy(R"predictMGPC($(epmodel),$(X_test))$prob"))
saveproba(py_ep,y_test,σ,"ep",false)
AUC_ep = 0#multiclassAUC(y_test,py_ep)
println("Expected model accuracy is $(gpflowacc(y_test,py_ep)), loglike : $(gpflowloglike(y_test,py_ep)) and AUC $(AUC_ep) in $t_ep s")
ep_map = epcallbackplot(epmodel,1,"Heaviside")
Plots.savefig(ep_map,"../plotslikelihood/contour_ep_σ$σ.pdf")
ep_map = title!(ep_map,"Heaviside")
ep_metrics = deepcopy(metrics)
ep_kerparams = deepcopy(kerparams)
ep_elbo = deepcopy(elbos)
AUC_ep = 0
# ECE_ep, MCE_ep, cal_ep, calh_ep, conf_ep= calibration(y_test,py_ep,nBins=nBins,plothist=true,plotline=true,plotconf=true,gpflow=true,meanonly=true,threshold=2)
# Plots.savefig(cal_ep,"../plotslikelihood/cal_line_ep_σ$σ.pdf")
# Plots.savefig(calh_ep,"../plotslikelihood/cal_hist_ep_σ$σ.pdf")
# Plots.savefig(conf_ep,"../plotslikelihood/cal_const_ep_σ$σ.pdf")

## Plotting part
pmet_alsm = plot(alsm_metrics,title="Aug. LogSoftMax",markersize=0.0,linewidth=2.0)
pmet_alsm = hline!(pmet_alsm,[bayes_error],lab="",line=(2.0,:red))
pmet_lsm = plot(lsm_metrics,title="LogSoftMax",markersize=0.0,linewidth=2.0)
pmet_lsm = hline!(pmet_lsm,[bayes_error],lab="",line=(2.0,:red))
pmet_sm = plot(sm_metrics,title="SoftMax",markersize=0.0,linewidth=2.0)
pmet_sm = hline!(pmet_sm,[bayes_error],lab="",line=(2.0,:red))
pmet_rm = plot(rm_metrics,title="RobustMax",markersize=0.0,linewidth=2.0)
pmet_rm = hline!(pmet_rm,[bayes_error],lab="",line=(2.0,:red))
# met_lims = (min(ylims(pmet_alsm)[1],ylims(pmet_lsm)[1],ylims(pmet_sm)[1],ylims(pmet_rm)[1]),max(ylims(pmet_alsm)[2],ylims(pmet_lsm)[2],ylims(pmet_sm)[2],ylims(pmet_rm)[2]))

pker_alsm = plot(alsm_kerparams,title="Aug. LogSoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
pker_lsm = plot(lsm_kerparams,title="LogSoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
pker_sm = plot(sm_kerparams,title="SoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
pker_rm = plot(rm_kerparams,title="RobustMax",yaxis=:log,markersize=0.0,linewidth=2.0)
# ker_lims = (min(ylims(pker_alsm)[1],ylims(pker_lsm)[1],ylims(pker_sm)[1],ylims(pker_rm)[1]),max(ylims(pker_alsm)[2],ylims(pker_lsm)[2],ylims(pker_sm)[2],ylims(pker_rm)[2]))

pelbo_alsm = plot(lsm_elbo,title="Aug. LogSoftMax",markersize=0.0,linewidth=2.0)
pelbo_lsm = plot(lsm_elbo,title="LogSoftMax",markersize=0.0,linewidth=2.0)
pelbo_sm = plot(sm_elbo,title="SoftMax",markersize=0.0,linewidth=2.0)
pelbo_rm = plot(rm_elbo,title="RobustMax",markersize=0.0,linewidth=2.0)
# elbo_lims = (min(ylims(pelbo_alsm)[1],ylims(pelbo_lsm)[1],ylims(pelbo_sm)[1],ylims(pelbo_rm)[1]),max(ylims(pelbo_alsm)[2],ylims(pelbo_lsm)[2],ylims(pelbo_sm)[2],ylims(pelbo_rm)[2]))

pmet = plot(pmet_alsm,pmet_lsm,pmet_sm,pmet_rm,link=:all)
pker = plot(pker_alsm,pker_lsm,pker_sm,pker_rm,link=:all)
pelbo = plot(pelbo_alsm,pelbo_lsm,pelbo_sm,pelbo_rm,link=:y)
lsm_map = callbackplot(lsmmodel,1,"Logistic-Softmax")
sm_map = callbackplot(smmodel,1,"Softmax")
rm_map = gpflowcallbackplot(rmmodel,1,"Robust-Max")
ep_map = epcallbackplot(epmodel,1,"Heaviside")

pmap = Plots.plot(sm_map,lsm_map,rm_map,ep_map,layout=(1,4),size=(1953,475))
Plots.savefig(pmap,"Contours_$σ.png")
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
Plots.savefig(pmet,"resultslikelihood/metrics_noise$(σ).png")
Plots.savefig(pmetfin,"resultslikelihood/metricsfinal_noise$(σ).png")
Plots.savefig(pmap,"resultslikelihood/plot_noise$(σ).png")
Plots.savefig(pelbo,"resultslikelihood/elbo_noise$(σ).png")
Plots.savefig(pker,"resultslikelihood/kernel_params_noise$(σ).png")
writedlm("resultslikelihood/results_$σ.txt",hcat([acc(y_test,y_alsm),loglike(y_test,py_alsm),mean(ECE_alsm),mean(MCE_alsm)],[acc(y_test,y_lsm),loglike(y_test,py_lsm),mean(ECE_lsm),mean(MCE_lsm)],[acc(y_test,y_sm),loglike(y_test,py_sm),mean(ECE_sm),mean(MCE_sm)],[gpflowacc(y_test,py_rm),gpflowloglike(y_test,py_rm),mean(ECE_rm),mean(MCE_rm)],[gpflowacc(y_test,py_ep),gpflowloglike(y_test,py_ep),mean(ECE_ep),mean(MCE_ep)]))
