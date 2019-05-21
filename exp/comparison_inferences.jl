using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using HDF5
using ValueHistories
using Plots
using LinearAlgebra
using MLDataUtils
const AGP = AugmentedGaussianProcesses
include("metrics.jl")
pyplot()
clibrary(:cmocean)
seed!(42)
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
f = "vehicle"
X = vcat(h5read("../data/"*f*".h5","data/X_train"),h5read("../data/"*f*".h5","data/X_test"))
y = vcat(h5read("../data/"*f*".h5","data/y_train"),h5read("../data/"*f*".h5","data/y_test"))
# size(X)
# data = h5read("../data/"*f*".h5","data")
# X = data[:,1:end-1]; y=data[:,end]

(X,y),(X_test,y_test) = MLDataUtils.splitobs((X,y),at=0.66,obsdim=1)
N_dim=size(X,2)


##
function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X,dims=1)
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
l = sqrt(initial_lengthscale(X))
variance=50.0
# kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim,variance=1.0)
kernel = RBFKernel(l,variance=variance)
# setfixed!(kernel.fields.lengthscales)
# setfixed!(kernel.fields.variance)
nBins = 10
autotuning = !true


### AUG. LOGISTIC SOFTMAX
amodel = VGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticVI(ϵ=1e-20),verbose=2,Autotuning=!true,IndependentPriors=true)
t_a = @elapsed train!(amodel,iterations=600)

y_a = proba_y(amodel,X_test)
μ_a,σ_a = predict_f(amodel,X_test,covf=true)
y_a = AGP.compute_proba(amodel.likelihood,μ_a,σ_a)
println("Augmented accuracy = : $(mean(predict_y(amodel,X_test).==y_test))")
# kernel = deepcopy(amodel.kernel[1])


## LOGISTIC SOFTMAX
emodel = VGP(X,y,kernel,LogisticSoftMaxLikelihood(),MCIntegrationVI(optimizer=VanillaGradDescent(η=0.01)),verbose=2,Autotuning=autotuning,IndependentPriors=!true)
t_e = @elapsed train!(emodel,iterations=400)

global y_e = proba_y(emodel,X_test)
global μ_e,σ_e = predict_f(emodel,X_test,covf=true)
println("Expec accuracy = $(mean(predict_y(emodel,X_test).==y_test))")
## GIBBS Sampling
gmodel = VGP(X,y,kernel,LogisticSoftMaxLikelihood(),GibbsSampling(nBurnin=100,samplefrequency=1,ϵ=1e-20),verbose=2,Autotuning=autotuning,IndependentPriors=!true)
t_g = @elapsed train!(gmodel,iterations=1000)
# @profiler train!(gmodel,iterations=200);
global y_g = proba_y(gmodel,X_test,nSamples=100)
global μ_g,σ_g = predict_f(gmodel,X_test,covf=true)
global y_g2 = AGP.compute_proba(gmodel.likelihood,μ_g,σ_g)
global μ_g2,σ_g2 = copy(μ_g),copy(σ_g)
println("Gibbs accuracy = : $(mean(predict_y(gmodel,X_test).==y_test))")
calibration(y_test,y_g,plothist=true,plotline=true)
## Plotting part

models = [:g,:e,:a,:g2]
labels = Dict(:g=>"Gibbs",:a=>"Augmented VI",:e=>"VI",:g2=>"Variational Gibbs")
iter = 1
function adaptlims(p,plims)
    x = xlims(p)
    y = ylims(p)
    plims[1] = min(min(x[1],y[1]),plims[1])
    plims[2] = max(max(x[2],y[2]),plims[2])
    xlims!(p,(plims[1],plims[2]))
    ylims!(p,(plims[1],plims[2]))
end
mulims = [Inf,-Inf]
siglims = [Inf,-Inf]

ps = []
testi = 4
markerarguments = (:auto,1.0,0.5,:black,stroke(0))
for i in 1:1
    for j in i+1:length(models)
        p_y = Plots.plot(vec(Matrix(eval(Symbol("y_",models[i]))[testi:testi])),vec(Matrix(eval(Symbol("y_",models[j]))[testi:testi])),yaxis=(labels[models[j]],(0,1),font(20)),t=:scatter,lab="",title="p",xaxis=(labels[models[i]],(0,1),font(20)),marker=markerarguments)
        Plots.plot!(p_y,x->x,lab="",tickfontsize=17)
        p_μ = Plots.plot(vcat(eval(Symbol("μ_",models[i]))[testi:testi]...),vcat(eval(Symbol("μ_",models[j]))[testi:testi]...),t=:scatter,lab="",title="μ",xaxis=(labels[models[i]],font(20)),marker=markerarguments)
        adaptlims(p_μ,mulims)
        p_σ = Plots.plot(vcat(eval(Symbol("σ_",models[i]))[testi:testi]...),vcat(eval(Symbol("σ_",models[j]))[testi:testi]...),t=:scatter,lab="",title="σ²",yaxis=:log,xaxis=(labels[models[i]],font(20),:log),marker=markerarguments)
        adaptlims(p_σ,siglims)
        # Plots.plot!(p_σ,x->x,lab="",tickfontsize=12)
        Plots.push!(ps,Plots.plot(p_y,p_μ,p_σ,layout=(1,3)))#,title="$(models[i]) vs $(models[j])"))
    end
    for j in 1:3
        xlims!(ps[j][2],(mulims[1],mulims[2]))
        ylims!(ps[j][2],(mulims[1],mulims[2]))
        Plots.plot!(ps[j][2],mulims,x->x,lab="",tickfontsize=17)
        xlims!(ps[j][3],(siglims[1],siglims[2]))
        ylims!(ps[j][3],(siglims[1],siglims[2]))
        Plots.plot!(ps[j][3],siglims,x->x,lab="",tickfontsize=17)
    end
end

# display(Plots.plot(ps...,layout=(2,1),dpi=300,size=(937,500)))
display(Plots.plot(ps...,layout=(3,1),dpi=300,size=(937,800)))
Plots.savefig("../plotsinference/gibbs comparison_$(f)_v$(getvariance(kernel)).pdf")
