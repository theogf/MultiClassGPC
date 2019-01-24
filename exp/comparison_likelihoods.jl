using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using PyCall
using ValueHistories
using Plots
using LinearAlgebra
pyplot()
seed!(42)
@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp
N_data = 300
N_class = 3
N_test = 50
N_grid = 50
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
N_iterations = 200

for c in 1:N_class
    global centers = rand(Uniform(-1,1),N_class,N_dim)*0.6
    global variance = 0.7*1/N_class*ones(N_class)#rand(Gamma(1.0,0.5),150)
end

X = zeros(N_data,N_dim)
y = sample(1:N_class,N_data)
for i in 1:N_data
    X[i,:] = rand(MvNormal(centers[y[i],:],variance[y[i]]))
end

xmin = minimum(X)*1.1; xmax = maximum(X)*1.1
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
x_grid = range(xmin,length=N_grid,stop=xmax)
X_grid = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])

metrics = MVHistory()
kerparams = MVHistory()
elbos = MVHistory()
anim  = Animation()
function callback(model,iter)
    if iter%2 !=0
        return
    end
    y_fgrid =  model.predict(X_grid)
    global py_fgrid = model.predictproba(X_grid)
    global cols = reshape([RGB(vec(convert(Array,py_fgrid[i,:]))[model.class_mapping]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
    col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global p1= plot(x_grid,x_grid,cols,t=:contour,colorbar=false)
    p1= plot!(x_grid,x_grid,reshape(y_fgrid,N_grid,N_grid),clims=[1.5,2.5],t=:contour,colorbar=false)
    p1=plot!(p1,X[:,1],X[:,2],color=col_doc[y],t=:scatter,lab="")
    p1=plot!(p1,model.inducingPoints[1][:,1],model.inducingPoints[1][:,2],color=:black,t=:scatter,lab="")
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

function callback2(model,iter)
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
            push!(kerparams,Symbol("l",i),p)
        end
        push!(kerparams,Symbol("v",i),getvariance(model.kernel[i]))
    end
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


elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()


lsmmodel = AugmentedGaussianProcesses.SparseLogisticSoftMaxMultiClass(X,y,verbose=2,ϵ=1e-20,kernel=kernel,optimizer=0.1,Autotuning=true,AutotuningFrequency=1,IndependentGPs=true,m=50)
Z = lsmmodel.inducingPoints
t_lsm = @elapsed lsmmodel.train(iterations=N_iterations,callback=callback2)

global py_lsm = lsmmodel.predictproba(X_test)
global y_lsm = lsmmodel.predict(X_test)
println("Expected model Accuracy is $(acc(y_test,y_lsm)) and loglike : $(loglike(y_test,py_lsm)) in $t_lsm s")
lsm_map = title!(callback(lsmmodel,2),"LogSoftMax")
lsm_metrics = deepcopy(metrics)
lsm_kerparams = deepcopy(kerparams)
lsm_elbo = deepcopy(elbos)

elbos = MVHistory()
metrics = MVHistory()
kerparams = MVHistory()
smmodel = AugmentedGaussianProcesses.SparseSoftMaxMultiClass(X,y,verbose=2,ϵ=1e-20,kernel=kernel,optimizer=0.01,Autotuning=true,AutotuningFrequency=1,IndependentGPs=true,m=50)
smmodel.inducingPoints = Z
t_sm = @elapsed smmodel.train(iterations=N_iterations,callback=callback2)

global py_sm = smmodel.predictproba(X_test)
global y_sm = smmodel.predict(X_test)
println("Augmented model Accuracy is $(acc(y_test,y_sm)) and loglike : $(loglike(y_test,py_sm)) in $t_sm")
sm_map= title!(callback(smmodel,2),"SoftMax")
sm_metrics = deepcopy(metrics)
sm_kerparams = deepcopy(kerparams)
sm_elbo = deepcopy(elbos)




### Plotting part

pmet_lsm = plot(lsm_metrics,title="LogSoftMax",markersize=0.0,linewidth=2.0)
pmet_sm = plot(sm_metrics,title="SoftMax",markersize=0.0,linewidth=2.0)
met_lims = (min(ylims(pmet_lsm)[1],ylims(pmet_sm)[1]),max(ylims(pmet_lsm)[2],ylims(pmet_sm)[2]))

pker_lsm = plot(lsm_kerparams,title="LogSoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
pker_sm = plot(sm_kerparams,title="SoftMax",yaxis=:log,markersize=0.0,linewidth=2.0)
ker_lims = (min(ylims(pker_lsm)[1],ylims(pker_sm)[1]),max(ylims(pker_lsm)[2],ylims(pker_sm)[2]))

pelbo_lsm = plot(lsm_elbo,title="LogSoftMax",markersize=0.0,linewidth=2.0)
pelbo_sm = plot(sm_elbo,title="SoftMax",markersize=0.0,linewidth=2.0)
elbo_lims = (min(ylims(pelbo_lsm)[1],ylims(pelbo_sm)[1]),max(ylims(pelbo_lsm)[2],ylims(pelbo_sm)[2]))

pmet = plot(ylims!(pmet_lsm,met_lims),ylims!(pmet_sm,met_lims))
pker = plot(ylims!(pker_lsm,ker_lims),ylims!(pker_sm,ker_lims))
pelbo = plot(ylims!(pelbo_lsm,elbo_lims),ylims!(pelbo_sm,elbo_lims))
pmap = plot(lsm_map,sm_map)
display(pmet)
display(pker)
display(pelbo)
display(pmap)
