using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using Makie, AbstractPlotting, Colors
using LinearAlgebra
using Plots: cgrad
cd(@__DIR__)
seed!(42)

## Parameters experiment
N_data = 1000; N_dim = 2
N_grid = 100;
m = 50;
function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X,dims=1)
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end


## Creating data
σ = 0.5; N_class = N_dim+1
## Create N_Class centers which are equidistant.
centers = zeros(N_class,N_dim)
for i in 1:N_dim
    centers[i,i] = 1
end
centers[end,:] = (1+sqrt(N_class))/N_dim*ones(N_dim)
centers./= sqrt(N_dim)

## Sample data randomly from this mixture
distr = [MvNormal(centers[i,:],σ) for i in 1:N_class]
X = zeros(Float64,N_data,N_dim)
y = zeros(Int64,N_data)
for i in 1:N_data
    y[i] = rand(1:N_class)
    X[i,:] = rand(distr[y[i]])
end
#Optimize the kernel for the GP
l = sqrt(initial_lengthscale(X))
kernel = RBFKernel([l],dim=N_dim,variance=1.0)

## Create a grid for predictions
xmin = minimum(X); xmax = maximum(X)
x_grid = range(xmin,length=N_grid,stop=xmax)
X_grid = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])

## Plotting parameters
scale = 1.0 #Relative z distance between layers
minalpha = 0.4 #Minimum transparency of the layers
offscale = 1.005 #Extra distance of the surrounding black rectangles (to avoid aliasing)
col_doc = [colorant"#252d4d",colorant"#dd4733",colorant"#eac675"] #Set of colors for the classes
# Gradients based on the colors going from white transparent to col
grads = [cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[1],1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[2],1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[3],1)])]




ptrans(i,n) = i/n


## First phase to move the latent GP rectangles out
function init_phase(io,scene,n)
    for i in 1:n
        scene[3][:color] = RGBA.(RGB.(scene[3].color[]),ptrans(i,n))
        pixelchild[2][:color] = RGBA.(colorant"black",ptrans(i,n))
        for j in 1:3
            Makie.translate!(scene[5+(j-1)*2],Tuple(vcat(Float32.(ptrans(i,n)*shifts[j][1:2]),Float32(-ptrans(i,n)*(4-j)*scale))))
            scene[5+(j-1)*2][:color] = RGBA(RGB(scene[5+(j-1)*2][:color][]),ptrans(i,n))
        end
        recordframe!(io)
    end
end


## Second phase during training where the latent GPs are trained
function livemakie(io::VideoStream,scene)
    function callback(model,iter)
        if iter%5 == 0
            μ_fgrid = predict_f(model,X_grid,covf=false) #Make latent predictions on the grid
            for i in 1:3
                μ = μ_fgrid[collect(values(sort(model.likelihood.ind_mapping)))][i]
                μ = (μ.-minimum(μ))/(maximum(μ)-minimum(μ))
                int_cols = reshape(getindex.([grads[i]],μ),N_grid,N_grid)' #Convert this predictions into colors
                scene[4+(i-1)*2][:color] = int_cols
            end
            AbstractPlotting.force_update!()
            recordframe!(io)
        end
    end
end

## Last phase where recangles are merged
function final_phase(io,model,n)
    py_fgrid = Matrix(proba_y(model,X_grid))[:,collect(values(sort(model.likelihood.ind_mapping)))] #Make likelihood prediction of the grid
    cols = reshape([RGBAf0(sum(py_fgrid[i,:].*col_doc)) for i in 1:N_grid*N_grid],N_grid,N_grid)
    saved_points = []
    for i in 1:3
        push!(saved_points,scene[5+(i-1)*2][1][])
    end
    ## Move rectangles in the middle and reduce opacity
    for j in 1:n/2
        for i in 1:3
            Makie.translate!(scene[4+(i-1)*2],Tuple(vcat(Float32.(-ptrans(j,n/2)*shifts[i][1:2]),[0f0])))
            Makie.translate!(scene[5+(i-1)*2],Tuple(vcat(Float32.((1.0.-ptrans(j,n/2))*shifts[i][1:2]),[Float32(-(4-i)*scale)])))
            pixelchild[2][:color] = RGBA.(colorant"black",1.0-ptrans(j,n/2))

        end
        recordframe!(io)
    end
    ## Make rectangles overlap and make them invisible in the middle and reduce opacity
    for j in 1:n/2
        for i in 1:3
            Makie.translate!(scene[4+(i-1)*2],Tuple(vcat(-Float32.(shifts[i][1:2]),Float32.(-ptrans(j,n/2)*shifts[i][3]))))
            Makie.translate!(scene[5+(i-1)*2],Tuple(vcat([0f0,0f0],Float32(-(4-i)*scale - ptrans(j,n/2)*shifts[i][3]))))
            pixelchild[3][:color] = RGBA.(colorant"black",ptrans(j,n/2))

        end
        recordframe!(io)
    end
    for i in 1:3
        scene[4+(i-1)*2][:visible] = false
        scene[5+(i-1)*2][:visible] = false
    end
    ## Make the final layer visible for some time
    scene[end-1][:color] = copy(transpose(cols))
    scene[end][:visible] = true
    for j in 1:n
        recordframe!(io)
    end
    #
end

width = xmax-xmin
shifts = [[width/sqrt(2),width/sqrt(2),-scale],[-width/sqrt(2),width/sqrt(2),0.0],[width/sqrt(2),-width/sqrt(2),scale]] #Full Movements  of the rectangles


## Initialize the scene with all its components (needed to be done before animations)
model = SVGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticSVI(10,optimizer=InverseDecay(τ=50)),m,verbose=2,IndependentPriors=!true)
scene = Scene()
Makie.scatter!([1,0,0],[0,1,0],[0,0,1],visible=false) #Setup to have 3D plots
Makie.scatter!(X[:,1],X[:,2],scale*(model.nLatent+1)*ones(size(X,1)),color=RGBA.(col_doc[y],0.0),lab="",markerstrokewidth=0.1,transparency=true,shading=false) #Plot Data
for i in 1:model.nLatent
    Makie.surface!(collect(x_grid).+shifts[i][1],collect(x_grid).+shifts[i][2],scale*i*ones(N_grid,N_grid),color=reshape(fill(RGBA(1,1,1,0),N_grid^2),N_grid,N_grid)',shading=false)
    Makie.lines!([xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],scale*(model.nLatent+1)*ones(5),lab="",color=RGBA(0,0,0,0),linewidth=2.0,overdraw=false)
end
Makie.lines!([xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],scale*(model.nLatent+1)*ones(5),lab="",color=:black,linewidth=2.0)
Makie.surface!(collect(x_grid),collect(x_grid),2*scale*ones(N_grid,N_grid),color=zeros(N_grid,N_grid)*RGBA(1,1,1,0),lab="",shading=false)#,size=(600,2000))#,colorbar=false,framestyle=:none,,dpi=dpi)
Makie.lines!([xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],2*scale*ones(5),lab="",color=RGBA(0,0,0,1),linewidth=2.0,shading=false,visible=false)

## Whole setup to get text to appear flat to the camera
pixelchild = campixel(scene);
pixelchild.center=false
zoffset = scale*0.2
cam = camera(scene)
font = "CambriaMath"
fontsize = 30.0

positiondata = [Point3f0(xmin,xmax,scale*(model.nLatent+1))+zoffset]
pixeldata_position = lift(cam.projectionview, cam.resolution) do pv, res
    AbstractPlotting.project.((pv,), (res,), positiondata)
end
annotations!(pixelchild, ["data"], pixeldata_position, raw = true, font = font,textsize=fontsize)
positionpf = [Point3f0(xmin,xmin+width/2,scale*(2) +zoffset)+shifts[1],
              Point3f0(xmin,xmin+width/2,scale*(2.5) +zoffset)+shifts[2],
              Point3f0(xmin+width,xmin-width/3,scale*(2.5) +zoffset)+shifts[3]]
pixelpf_position = lift(cam.projectionview, cam.resolution) do pv, res
    AbstractPlotting.project.((pv,), (res,), positionpf)
end
annotations!(pixelchild, ["p(f ¹|data)","p(f ²|data)","p(f ³|data)"], pixelpf_position, raw = true,color=RGBA(0,0,0,0),font=font,textsize=fontsize)
positionpy = [Point3f0(xmin,xmax,scale*2)+zoffset]
pixelpy_position = lift(cam.projectionview, cam.resolution) do pv, res
    AbstractPlotting.project.((pv,), (res,), positionpy)
end
annotations!(pixelchild, ["p(y|data)"], pixelpy_position, raw = true,color=RGBA(0,0,0,0),font=font,textsize=fontsize)
scene


## Remove all grid and axis
scene[Axis][:showgrid] = (false,false,false)
scene[Axis][:showaxis] = (false,false,false)
scene[Axis][:ticks][:textsize] = 0
scene[Axis][:names][:axisnames] = ("","","")
## Finally run the whole recording
record(scene,"multi_class_gp.gif",framerate=15) do io
    init_phase(io,scene,15)
    train!(model,iterations=300,callback=livemakie(io,scene))
    final_phase(io,model,30)
end
