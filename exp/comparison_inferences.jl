using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using HDF5
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
cd(@__DIR__)

N_data = 1000
N_class = 3
N_test = 50
N_grid = 100
minx=-5.0
maxx=5.0
noise = 1.0

N_iterations = 50
m = 20
art_noise = 0.5
f = "glass"
# X = vcat(h5read("../data/"*f*".h5","data/X_train"),h5read("../data/"*f*".h5","data/X_test"))
# y = vcat(h5read("../data/"*f*".h5","data/y_train"),h5read("../data/"*f*".h5","data/y_test"))
# size(X)
data = h5read("../data/"*f*".h5","data")
X = data[:,1:end-1]; y=data[:,end]

X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
N_dim=size(X,2)


##
function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X')
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
l = sqrt(initial_lengthscale(X))
variance=10.0
# kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim,variance=1.0)
kernel = AugmentedGaussianProcesses.RBFKernel(l,variance=variance)
# setfixed!(kernel.fields.lengthscales)
# setfixed!(kernel.fields.variance)
nBins = 10
autotuning = !true


### AUG. LOGISTIC SOFTMAX
amodel = VGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),AnalyticInference(ϵ=1e-20),verbose=2,Autotuning=autotuning,IndependentPriors=!true)
t_a = @elapsed train!(amodel,iterations=100)

global y_a = proba_y(amodel,X_test)
global μ_a,σ_a = predict_f(amodel,X_test,covf=true)
println("Augmented accuracy = : $(mean(predict_y(amodel,X_test).==y_test))")


## LOGISTIC SOFTMAX
using GradDescent
emodel = VGP(X,y,kernel,LogisticSoftMaxLikelihood(),NumericalInference(:mcmc,ϵ=1e-20,optimizer=VanillaGradDescent(η=0.01)),verbose=2,Autotuning=autotuning,IndependentPriors=!true)
t_e = @elapsed train!(emodel,iterations=400)

global y_e = proba_y(emodel,X_test)
global μ_e,σ_e = predict_f(emodel,X_test,covf=true)
println("Expec accuracy = $(mean(predict_y(emodel,X_test).==y_test))")
## GIBBS Sampling
gmodel = VGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),GibbsSampling(nBurnin=50,samplefrequency=10,ϵ=1e-20),verbose=2,Autotuning=autotuning,IndependentPriors=!true)
t_g = @elapsed train!(gmodel,iterations=10000)

global y_g = proba_y(gmodel,X_test)
global μ_g,σ_g = predict_f(gmodel,X_test,covf=true)
println("Gibbs accuracy = : $(mean(predict_y(gmodel,X_test).==y_test))")
calibration(y_test,y_g,plothist=true,plotline=true)
## Plotting part

models = [:g,:e,:a]
labels = Dict(:g=>"Gibbs",:a=>"Augmented VI",:e=>"VI")
iter = 1
function adaptlims(p)
    x = xlims(p)
    y = ylims(p)
    amin = min(x[1],y[1])
    amax = max(x[2],y[2])
    xlims!(p,(amin,amax))
    ylims!(p,(amin,amax))
end

ps = []
markerarguments = (:auto,1.0,0.5,:black,stroke(0))
for i in 1:1
    for j in i+1:3
        p_y = Plots.plot(vec(Matrix(eval(Symbol("y_",models[i])))),vec(Matrix(eval(Symbol("y_",models[j])))),yaxis=(labels[models[j]],(0,1),font(20)),t=:scatter,lab="",title="p",xaxis=(labels[models[i]],(0,1),font(20)),marker=markerarguments)
        Plots.plot!(p_y,x->x,lab="",tickfontsize=17)
        p_μ = Plots.plot(vcat(eval(Symbol("μ_",models[i]))...),vcat(eval(Symbol("μ_",models[j]))...),t=:scatter,lab="",title="μ",xaxis=(labels[models[i]],font(20)),marker=markerarguments)
        adaptlims(p_μ)
        Plots.plot!(p_μ,x->x,lab="",tickfontsize=17)
        p_σ = Plots.plot(vcat(eval(Symbol("σ_",models[i]))...),vcat(eval(Symbol("σ_",models[j]))...),t=:scatter,lab="",title="σ²",yaxis=:log,xaxis=(labels[models[i]],font(20),:log),marker=markerarguments)
        adaptlims(p_σ)
        Plots.plot!(p_σ,x->x,lab="",tickfontsize=12)
        Plots.push!(ps,Plots.plot(p_y,p_μ,p_σ,layout=(1,3)))#,title="$(models[i]) vs $(models[j])"))
    end
end

display(Plots.plot(ps...,layout=(2,1),dpi=300,size=(937,500)))
Plots.savefig("../plotsinference/gibbs comparison_$(f)_v$(variance).pdf")
