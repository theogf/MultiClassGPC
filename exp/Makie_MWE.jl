using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using ValueHistories
using Plots
using Makie, AbstractPlotting
using LinearAlgebra
using DelimitedFiles
using MLDataUtils
cd(@__DIR__)
pyplot()
seed!(42)

## Parameters experiment
N_data = 1000
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
## Creating data
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
(X,y),(X_test,y_test) = splitobs((X,y),at=0.66,obsdim=1)
x_grid = range(xmin,length=N_grid,stop=xmax)
X_grid = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])
# global col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)] #RGB Colros
# global col_doc = [colorant"#389826",colorant"#9558B2",colorant"#CB3C33"] #Julia Colors
# global col_doc = [colorant"#252d4d",colorant"#dd4733",colorant"#c8a136"] #Ronja Colors
global col_doc = [colorant"#252d4d",colorant"#dd4733",colorant"#eac675"] #Ronja Colors
# global col_doc = [colorant"#B82000",colorant"#A5FF00",colorant"#0443CC"] #Custom Colors
global scale = 1.0
tsize = 0.8
minalpha = 0.4
offscale = 1.005

function callbackmakie(model)
    global y_fgrid = predict_y(model,X_grid)
    global py_fgrid = Matrix(proba_y(model,X_grid))[:,collect(values(sort(model.likelihood.ind_mapping)))]
    global μ_fgrid = predict_f(model,X_grid,covf=false)
    global cols = reshape([sum(py_fgrid[i,:].*col_doc) for i in 1:N_grid*N_grid],N_grid,N_grid)
    global scene
    Makie.scatter!(scene,[1,0,0],[0,1,0],[0,0,1],color=RGBA(1,1,1,0)) #For 3D plots
    Makie.scatter!(scene,X[:,1],X[:,2],scale*(model.nLatent+1)*ones(size(X,1)),color=col_doc[y],lab="",markerstrokewidth=0.1,transparency=true,shading=false)
    Makie.surface!(scene,collect(x_grid),collect(x_grid),zeros(N_grid,N_grid),grid=:hide,color=cols',lab="",shading=false)#,size=(600,2000))#,colorbar=false,framestyle=:none,,dpi=dpi)
    offscale = 1.005
    Makie.lines!(scene,[xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],zeros(5),lab="",color=:black,linewidth=2.0,shading=false)
    tsize = 0.8
    minalpha = 0.4
    grads = [cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[1],1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[2],1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[3],1)])]
    # grads = [cgrad([RGBA(1,1,1,minalpha),RGBA(1,0,0,1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(0,1,0,1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(0,0,1,1)])]
    # Makie.text!(scene,"p(y|D)",position=(xmin,xmax,0.0),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    sub = ["₃","₂","₁"]
    for i in 1:model.nLatent
        μ = μ_fgrid[collect(values(sort(model.likelihood.ind_mapping)))][i]
        μ = (μ.-minimum(μ))/(maximum(μ)-minimum(μ))
        int_cols = getindex.([grads[i]],μ)
        Makie.surface!(scene,collect(x_grid),collect(x_grid),scale*i*ones(N_grid,N_grid),color=reshape(int_cols,N_grid,N_grid)',shading=false)
        Makie.lines!(scene,[xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],scale*i*ones(5),lab="",color=:black,linewidth=2.0,overdraw=false)
        # Makie.text!(scene,"p(f"*sub[i]*"|D)",position = (xmin,xmax,scale*i),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    end

    Makie.lines!(scene,[xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],scale*(model.nLatent+1)*ones(5),lab="",color=:black,linewidth=2.0)
    scene[Axis][:showgrid] = (false,false,false)
    scene[Axis][:showaxis] = (false,false,false)
    scene[Axis][:ticks][:textsize] = 0
    scene[Axis][:names][:axisnames] = ("","","")
    # Makie.text!(scene,"data",position = (xmin,xmax,scale*(model.nLatent+1)),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    scene.center=false
    return scene
end

function init_callbackmakie(model,scene)
    global y_fgrid = predict_y(model,X_grid)
    global py_fgrid = Matrix(proba_y(model,X_grid))[:,collect(values(sort(model.likelihood.ind_mapping)))]
    global μ_fgrid = predict_f(model,X_grid,covf=false)
    global cols = reshape([sum(py_fgrid[i,:].*col_doc) for i in 1:N_grid*N_grid],N_grid,N_grid)
    # Makie.surface!(scene,collect(x_grid),collect(x_grid),zeros(N_grid,N_grid),grid=:hide,color=cols',lab="",shading=false)#,size=(600,2000))#,colorbar=false,framestyle=:none,,dpi=dpi)
    # Makie.lines!(scene,[xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],zeros(5),lab="",color=:black,linewidth=2.0,shading=false)
    # Makie.text!(scene,"p(y|D)",position=(xmin,xmax,0.0),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    grads = [cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[1],1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[2],1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[3],1)])]
    sub = ["₃","₂","₁"]
    for i in 1:model.nLatent
        μ = μ_fgrid[collect(values(sort(model.likelihood.ind_mapping)))][i]
        μ = (μ.-minimum(μ))/(maximum(μ)-minimum(μ))
        int_cols = getindex.([grads[i]],μ)
        Makie.surface!(collect(x_grid),collect(x_grid),scale*i*ones(N_grid,N_grid),color=reshape(int_cols,N_grid,N_grid)',shading=false)
        Makie.lines!([xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],scale*i*ones(5),lab="",color=:black,linewidth=2.0,overdraw=false)
        # Makie.text!(scene,"p(f"*sub[i]*"|D)",position = (xmin,xmax,scale*i),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    end

    Makie.lines!([xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],scale*(model.nLatent+1)*ones(5),lab="",color=:black,linewidth=2.0)
    # Makie.text!(scene,"data",position = (xmin,xmax,scale*(model.nLatent+1)),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    scene.center=false
    AbstractPlotting.force_update!()
end



function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X,dims=1)
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
l = sqrt(initial_lengthscale(X))

kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim,variance=1.0)
# kernel = AugmentedGaussianProcesses.RBFKernel(l,variance=10.0)
# setfixed!(kernel.fields.lengthscales)
# setfixed!(kernel.fields.variance)
nBins = 10
autotuning = true
scene = Scene(resolution=(300,300))

ptrans(i,n) = i/n

function init_phase(io,scene,n)
    for i in 1:n
        scene[3][:color] = RGBA.(RGB.(scene[3].color[]),ptrans(i,n))
        pixelchild[2][:color] = RGBA.(colorant"black",ptrans(i,n))
        # scene[14][:color] = RGBA.(RGB.(scene[3].color[]),ptrans(i,n))
        for j in 1:3
            Makie.translate!(scene[5+(j-1)*2],Tuple(vcat(Float32.(ptrans(i,n)*shifts[j][1:2]),Float32(-ptrans(i,n)*(4-j)*scale))))
            scene[5+(j-1)*2][:color] = RGBA(RGB(scene[5+(j-1)*2][:color][]),ptrans(i,n))
        end
        recordframe!(io)
    end
end
function livemakie(io::VideoStream,scene)
    function callback(model,iter)
        if iter%5 == 0
            μ_fgrid = predict_f(model,X_grid,covf=false)
            grads = [cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[1],1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[2],1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(col_doc[3],1)])]
            for i in 1:3
                μ = μ_fgrid[collect(values(sort(model.likelihood.ind_mapping)))][i]
                μ = (μ.-minimum(μ))/(maximum(μ)-minimum(μ))
                global int_cols = reshape(getindex.([grads[i]],μ),N_grid,N_grid)'
                scene[4+(i-1)*2][:color] = int_cols
            end
            AbstractPlotting.force_update!()
            recordframe!(io)
        end
    end
end
function final_phase(io,model,n)
    py_fgrid = Matrix(proba_y(model,X_grid))[:,collect(values(sort(model.likelihood.ind_mapping)))]
    global cols = reshape([RGBAf0(sum(py_fgrid[i,:].*col_doc)) for i in 1:N_grid*N_grid],N_grid,N_grid)
    global saved_points = []
    for i in 1:3
        push!(saved_points,scene[5+(i-1)*2][1][])
    end
    for j in 1:n/2
        for i in 1:3
            Makie.translate!(scene[4+(i-1)*2],Tuple(vcat(Float32.(-ptrans(j,n/2)*shifts[i][1:2]),[0f0])))
            Makie.translate!(scene[5+(i-1)*2],Tuple(vcat(Float32.((1.0.-ptrans(j,n/2))*shifts[i][1:2]),[Float32(-(4-i)*scale)])))
            # @show Tuple(vcat(Float32.(-ptrans(j,n/2)*shifts[i][1:2]),[0f0]))
            # push!(scene[5+(i-1)*2][1],saved_points[i] .- [Point3f0(Float32(shifts[i][1]),Float32(shifts[i][2]),Float32(shifts[i][3]))*Float32(ptrans(j,n))])
            pixelchild[2][:color] = RGBA.(colorant"black",1.0-ptrans(j,n/2))
            # pixelchild[3][:color] = RGBA.(colorant"black",ptrans(j,n))

        end
        # AbstractPlotting.force_update!()
        recordframe!(io)
    end
    for j in 1:n/2
        for i in 1:3
            Makie.translate!(scene[4+(i-1)*2],Tuple(vcat(-Float32.(shifts[i][1:2]),Float32.(-ptrans(j,n/2)*shifts[i][3]))))
            Makie.translate!(scene[5+(i-1)*2],Tuple(vcat([0f0,0f0],Float32(-(4-i)*scale - ptrans(j,n/2)*shifts[i][3]))))
            # push!(scene[5+(i-1)*2][1],saved_points[i] .- [Point3f0(Float32(shifts[i][1]),Float32(shifts[i][2]),Float32(shifts[i][3]))*Float32(ptrans(j,n))])
            # pixelchild[2][:color] = RGBA.(colorant"black",1.0-ptrans(j+n/2,n))
            pixelchild[3][:color] = RGBA.(colorant"black",ptrans(j,n/2))

        end
        recordframe!(io)
    end
    for i in 1:3
        scene[4+(i-1)*2][:visible] = false
        scene[5+(i-1)*2][:visible] = false
    end
    scene[end-1][:color] = copy(transpose(cols))
    scene[end][:visible] = true
    for j in 1:n
        recordframe!(io)
    end
    #
end
model = SVGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticVI(),m,verbose=2,Autotuning=autotuning,IndependentPriors=!true)
μ_fgrid = predict_f(model,X_grid,covf=false)
grads = [cgrad([RGBA(1,1,1,0),RGBA(1,1,1,0)]),cgrad([RGBA(1,1,1,0),RGBA(1,1,1,0)]),cgrad([RGBA(1,1,1,0),RGBA(1,1,1,0)])]
width = xmax-xmin
shifts = [[width/sqrt(2),width/sqrt(2),-scale],[-width/sqrt(2),width/sqrt(2),0.0],[width/sqrt(2),-width/sqrt(2),scale]]
## AUG. LOGISTIC SOFTMAX
scene = Scene()
Makie.scatter!([1,0,0],[0,1,0],[0,0,1],visible=false) #For 3D plots
Makie.scatter!(X[:,1],X[:,2],scale*(model.nLatent+1)*ones(size(X,1)),color=RGBA.(col_doc[y],0.0),lab="",markerstrokewidth=0.1,transparency=true,shading=false)
for i in 1:model.nLatent
    μ = μ_fgrid[collect(values(sort(model.likelihood.ind_mapping)))][i]
    μ = (μ.-minimum(μ))/(maximum(μ)-minimum(μ))
    int_cols = getindex.([grads[i]],μ)
    Makie.surface!(collect(x_grid).+shifts[i][1],collect(x_grid).+shifts[i][2],scale*i*ones(N_grid,N_grid),color=reshape(int_cols,N_grid,N_grid)',shading=false)
    Makie.lines!([xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],scale*(model.nLatent+1)*ones(5),lab="",color=RGBA(0,0,0,0),linewidth=2.0,overdraw=false)
    # Makie.lines!([xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale].+shifts[i][1],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale].+shifts[i][2],scale*i*ones(5),lab="",color=RGBA(0,0,0,0),linewidth=2.0,overdraw=false)
    # Makie.text!(scene,"p(f"*sub[i]*"|D)",position = (xmin,xmax,scale*i),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
end
Makie.lines!([xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],scale*(model.nLatent+1)*ones(5),lab="",color=:black,linewidth=2.0)
Makie.surface!(collect(x_grid),collect(x_grid),2*scale*ones(N_grid,N_grid),color=zeros(N_grid,N_grid)*RGBA(1,1,1,0),lab="",shading=false)#,size=(600,2000))#,colorbar=false,framestyle=:none,,dpi=dpi)
Makie.lines!([xmin*offscale,xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale],[xmin*offscale,xmax*offscale,xmax*offscale,xmin*offscale,xmin*offscale],2*scale*ones(5),lab="",color=RGBA(0,0,0,1),linewidth=2.0,shading=false,visible=false)
if @isdefined bestcam
    scene.center=false
    update_cam!(scene,bestcam)
end

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

scene[Axis][:showgrid] = (false,false,false)
scene[Axis][:showaxis] = (false,false,false)
scene[Axis][:ticks][:textsize] = 0
scene[Axis][:names][:axisnames] = ("","","")
N_iterations = 300
record(scene,"test.gif",framerate=15) do io
    init_phase(io,scene,15)
    model = SVGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticSVI(10,optimizer=InverseDecay(τ=50)),m,verbose=2,Autotuning=autotuning,IndependentPriors=!true)
    train!(model,iterations=N_iterations,callback=livemakie(io,scene))
    final_phase(io,model,30)
end
# callbackmakie(model)
