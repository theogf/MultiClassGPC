using OMGP, LinearAlgebra, Distributions,Plots, LaTeXStrings, PyCall, DelimitedFiles
@pyimport sklearn.model_selection as sp
pyplot()
function logit(x)
    return 1.0./(1.0.+exp.(-x))
end
function mod_soft_max(σ::Vector{T},sumsig::T=zero(T)) where {T}
    return sumsig == 0 ? σ./(sum(σ)) : σ./sumsig
end
function get_Dataset(datasetname::String)
    println("Getting dataset")
    data = readdlm("../data/"*datasetname*".csv",',')
    # data = Matrix{Float64}(CSV.read("../data/"*datasetname*".csv",header=false))
    X = data[:,1:end-1]; y = floor.(Int64,data[:,end]);
    println("Dataset loaded")
    return (X,y,datasetname)
end
N_dim = 2
N_samples = 1500
N_test = 20
kernel = RBFKernel([5.0],dim=N_dim,variance=20.0)
noise = 1e-1
N_classes = 4
minx=-1.0
maxx=1.0
X = (rand(N_samples,N_dim)*(maxx-minx)).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
# X_test = rand(trunc_d,N_test^dim,dim)

X = rand(N_samples,N_dim).*2.0.-1.0

function sample_gaussian_process(X,noise,kernel)
    N = size(X,1)
    K = kernelmatrix(X,kernel)+noise*Diagonal{Float64}(I,N)
    return rand(MvNormal(zeros(N),K))
end

fs = [sample_gaussian_process(X,noise,kernel) for _ in 1:N_classes]
y = argmax.([getindex.(fs,i) for i in 1:N_samples])
# X_test = X[floor(Int64,N_samples*2/3):end,:]; y_test = y[floor(Int64,N_samples*2/3):end]
# X = X[1:floor(Int64,N_samples*2/3),:]; y = y[1:floor(Int64,N_samples*2/3)]

(X,y,datasetname ) = get_Dataset("letter")
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.3)
kernel = RBFKernel(5.0)


p=plot()
p=[scatter!(X[y.==i,1],X[y.==i,2],lab="y=$i") for i in 1:N_classes]
plot(p...)

model = SparseMultiClass(X,y,kernel=kernel,m=200,batchsize=100,VerboseLevel=2,Autotuning=true,IndependentGPs=false)
model.train(iterations=100)
println("model trained")
full_f_star,full_cov_f_star = OMGP.fstar(model,X_test)
logit_f = logit.(full_f_star)
OMGP.multiclasspredict(model,X_test[1:2,:],true)
y1,v2 = OMGP.multiclasspredict(model,X_test,false)
argmin(v2)
v2[1038]
OMGP.multiclasspredictproba(model,X_test[1:2,:],true)
# OMGP.multiclasspredictproba_cubature(model,X_test[1:2,:],true)
OMGP.multiclasspredictprobamcmc(model,X_test[1:2,:],5000)
t_base = @elapsed m_base = OMGP.multiclasspredict(model,X_test,true)
t_tay = @elapsed m_pred_tay,sig_pred_tay = OMGP.multiclasspredictproba(model,X_test,true)
# t_cub = @elapsed m_pred_cub,sig_pred_cub = OMGP.multiclasspredictproba_cubature(model,X_test,true)
t_mc = @elapsed m_pred_mc,sig_pred_mc =  OMGP.multiclasspredictprobamcmc(model,X_test,5000)

diff_base = [mean(abs.(hcat(m_base.-m_pred_mc...))) ,0.0]
diff_tay = [mean(abs.(hcat(m_pred_tay-m_pred_mc...))) ,mean(abs.(hcat(sig_pred_tay-sig_pred_mc...)))]
# diff_cub = [mean(abs.(hcat(m_pred_cub-m_pred_mc...))) ,mean(abs.(hcat(sig_pred_cub-sig_pred_mc...)))]
N_test = length(y_test)
println("Base : $(t_base/N_test) s, $diff_base,\nTaylor : $(t_tay/N_test) s, $diff_tay\n","\nMC : $(t_mc/N_test)")

n = size(X_test,1)
m_f,cov_f = OMGP.fstar(model,X_test)
m_f = hcat(m_f...)
m_f = [m_f[i,:] for i in 1:n]
cov_f = hcat(cov_f...)
cov_f = [cov_f[i,:] for i in 1:n]
stack_preds = Vector{Vector{Any}}(undef,n);
m_pred_mc = Vector{Vector{Float64}}(undef,n)
sig_pred_mc = Vector{Vector{Float64}}(undef,n)
NSamples = 10000
m_evol = Vector{Vector{Float64}}(undef,NSamples)
cov_evol = Vector{Vector{Float64}}(undef,NSamples)
for i in 944
    preds = Vector{Vector{Float64}}(undef,NSamples)
    for j in 1:NSamples
        samp = logit.(broadcast((m,cov)->rand(Normal(m,cov)),m_f[i],cov_f[i]))
        norm_sig = sum(samp)
        preds[j] = mod_soft_max(samp,norm_sig)
        m_evol[j]= mean(preds[1:j])
        cov_evol[j] = cov.([broadcast(x->x[k],preds[1:j]) for k in 1:model.K])
        if j%100 == 0
            println("$j/$NSamples samples")
        end
    end
    # m_pred_mc[i]=mean(preds)
    # sig_pred_mc[i]=cov.([broadcast(x->x[j],preds) for j in 1:model.K])
end

p1=plot(hcat(m_evol...)')
p2=plot(hcat(cov_evol...)')
plot(p1,p2)
if false
class = 1
    pmc = plot(x_test,x_test,reshape(getindex.(m_pred_mc,class),N_test,N_test),t=:contour,clims=[0,1],fill=true,cbar=true,lab="",title="MC Integration")
    ptaylor = plot(x_test,x_test,reshape(getindex.(m_pred,class),N_test,N_test),t=:contour,clims=[0,1],fill=true,cbar=true,lab="",title="2nd order Taylor expansion")
    plotcubature = plot(x_test,x_test,reshape(getindex.(m_pred_cub,class),N_test,N_test),t=:contour,clims=[0,1],fill=true,cbar=true,lab="",title="Adaptive Cubature")
    display(plot(pmc,ptaylor,plotcubature,layout=(1,3)))
end
