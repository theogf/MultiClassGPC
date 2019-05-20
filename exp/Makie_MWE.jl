using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using PyCall
using ValueHistories
using Plots
using Makie
using LinearAlgebra
using GradDescent
using DelimitedFiles
cd(@__DIR__)
pyplot()
clibrary(:cmocean)
seed!(42)
@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp


N_data = 500
N_class = 3
N_test = 50
N_grid = 100
minx=-5.0
maxx=5.0
noise = 1.0

N_dim=2
N_iterations = 500
m = 50
art_noise = 0.3
dpi=600
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

##
xmin = minimum(X); xmax = maximum(X)
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
x_grid = range(xmin,length=N_grid,stop=xmax)
X_grid = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])

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

alsmmodel = SVGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),AnalyticInference(),m,verbose=2,Autotuning=autotuning,IndependentPriors=!true)
Z = copy(alsmmodel.Z)
t_alsm = @elapsed train!(alsmmodel,iterations=N_iterations)
@profiler train!(alsmmodel,iterations=10)

callbackmakie(alsmmodel)
