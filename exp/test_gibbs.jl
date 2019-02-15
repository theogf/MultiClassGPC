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
    p1= plot!(x_grid,x_grid,reshape(y_fgrid,N_grid,N_grid),clims=(0,100),t=:contour,colorbar=false,color=:gray,levels=10)
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

function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X')
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
l = sqrt(initial_lengthscale(X))

# kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim,variance=10.0)
kernel = AugmentedGaussianProcesses.RBFKernel(l,variance=10.0)
# setfixed!(kernel.fields.lengthscales)
# setfixed!(kernel.fields.variance)
autotuning = true

### AUG. LOGISTIC SOFTMAX
elbos = MVHistory()
metrics = MVHistory()
lparams = MVHistory()
vparams = MVHistory()
params = MVHistory()

model = AugmentedGaussianProcesses.GibbsSamplerMultiClass(X,y,verbose=2,ϵ=1e-20,kernel=kernel,Autotuning=false,AutotuningFrequency=1,IndependentGPs=true)
model.train(iterations=400)

# alsmmodel.train(iterations=10)
# @profiler alsmmodel.train(iterations=10)
# @prof}iler AugmentedGaussianProcesses.Gradient_Expec(alsmmodel)
# @btime AugmentedGaussianProcesses.Gradient_Expec(alsmmodel)
global py_alsm = model.predictproba(X_test)
global y_alsm = model.predict(X_test)
# println("Expected model Accuracy is $(acc(y_test,y_alsm)) and loglike : $(loglike(y_test,py_alsm)) in $t_alsm s")
##
map = title!(callbackplot(model,2),"Hybrid LogSoftMax");
metrics = deepcopy(metrics)
params = deepcopy(params)

pmet = plot(metrics,title="Gibbs LogSoftMax",markersize=0.0,linewidth=2.0)

ppar = plot(params,title="Gibbs LogSoftMax",markersize=0.0,linewidth=2.0)

display(pmet)
display(ppar)
display(pelbo)
