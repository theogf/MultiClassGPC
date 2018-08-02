using PyCall
@pyimport TTGP
@pyimport sklearn.datasets as skdata
@pyimport sklearn.model_selection as skmodel
doPlot = false


N_data = 300
N_class = 3
N_test = 50
minx=-5.0
maxx=5.0
noise = 1.0
truthknown = false
doMCCompare = false
dolikelihood = false
println("$(now()): Starting testing multiclass")

function latent(X)
    return sqrt.(X[:,1].^2+X[:,2].^2)
end
dim=2
X = (rand(N_data,dim)*(maxx-minx))+minx
trunc_d = Truncated(Normal(0,3),minx,maxx)
X = rand(trunc_d,N_data,dim)
x_test = linspace(minx,maxx,N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
# X_test = rand(trunc_d,N_test^dim,dim)
y = min.(max.(1,floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))),N_class)
y_test =  min.(max.(1,floor.(Int64,latent(X_test))),N_class)

X,y = skdata.make_classification(n_samples=N_data,n_features=dim,n_classes=N_class,n_clusters_per_class=1,n_informative=dim,n_redundant=0)
y+=1
X,X_test,y,y_test = skmodel.train_test_split(X,y,test_size=0.33)


m=20
ind_points = KMeansInducingPoints(X,m,10)
batchsize=40
maxiter=1000
