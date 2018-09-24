using Distributions
using PyCall
using Plots
@pyimport gpflow
@pyimport tensorflow as tf
using Dates
using Distributions
using OMGP
@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp
N_samples = 1000
N_dim = 2
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
X = (rand(N_samples,N_dim)*(maxx-minx)).+minx
trunc_d = Truncated(Normal(0,3),minx,maxx)
X = rand(trunc_d,N_samples,N_dim)
x_test = range(minx,stop=maxx,length=N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
# X_test = rand(trunc_d,N_test^dim,dim)
y = min.(max.(1,floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))),N_class).-1
y_test =  min.(max.(1,floor.(Int64,latent(X_test))),N_class).-1
# y = y.-1; y_tes
# X,y = sk.make_classification(n_samples=N_samples,n_features=N_dim,n_classes=N_class,n_clusters_per_class=1,n_informative=N_dim,n_redundant=0)
# X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
#Test on the Iris dataet
# train = readdlm("data/iris-X")
# X = train[1:100,:]; X_test=train[101:end,:]
# test = readdlm("data/iris-y")
# y = test[1:100,:]; y_test=test[101:end,:]
# truthknown = false
#
# data = readdlm("data/Glass")
# # Dataset has been already randomized
# X = data[1:150,1:(end-1)]; y=data[1:150,end]
# X_test = data[151:end,1:(end-1)]; y_test=data[151:end,end]
#### Test on the mnist dataset
# X = readdlm("data/mnist_train")
# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/mnist_test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]
# println("$(now()): MNIST data loaded")
#
# ### Test on the artificial character dataset
# X = readdlm("data/artificial-characters-train")
# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/artificial-characters-test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]
# println("$(now()): Artificial Characters data loaded")
m=20
batchsize=40
iterations=10000
Z = KMeansInducingPoints(X,m,10)
l=1.0
kernel = gpflow.kernels[:RBF](N_dim,lengthscales=l,ARD=true)+gpflow.kernels[:White](N_dim,variance=noise)
model = gpflow.models[:SVGP](X, Float64.(reshape(y,(length(y),1))),num_latent=N_class,kern=kernel, likelihood=gpflow.likelihoods[:MultiClass](N_class), Z=Z)
LogArrays = Array{Any,1}()
iter_points = vcat(1:99,100:10:999,1000:100:9999)
function TestAccuracy(y_test, y_predic)
    score = 0
    for i in 1:length(y_test)
        if (argmax(y_predic[i,:])-1) == y_test[i]
            score += 1
        end
    end
    return score/length(y_test)
end
function LogLikelihood(y_test,y_predic)
    return [log(y_predic[i,y_t+1]) for (i,y_t) in enumerate(y_test)]
end
function pythonlogger(model,session,iter)
      if in(iter,iter_points)
          a = zeros(8)
          a[1] = time_ns()
          y_p = model[:predict_y](X_test)[1]
          best = [argmax(y_p[i,:]) for i in 1:size(y_test,1)]
          score = 0
          for i in 1:size(y_test,1)
              if best[i]==y_test[i]
                  score += 1
              end
          end
          loglike = LogLikelihood(y_test,y_p)
          a[2] = TestAccuracy(y_test,y_p)
          a[3] = mean(loglike)
          a[4] = median(loglike)
          a[5] = session[:run](model[:likelihood_tensor])
          # println("Iteration $(self[:i]) : Acc is $(a[2]), MedianL is $(a[4]), ELBO is $(a[5]) mean(θ) is $(mean(tm.Model[i][:kern][:rbf][:lengthscales][:value]))")
          a[6] = time_ns()
          a[7] = model[:kern][:kernels][1][:lengthscales][:value][1]
          a[8] = model[:kern][:kernels][1][:variance][:value][1]
          push!(LogArrays,a)
          # println((a[1]-LogArrays[1][1])*1e-9)
      end
end

function run_nat_grads_with_adam(model,iterations; ind_points_fixed=true, kernel_fixed =false, callback=nothing)
    # we'll make use of this later when we use a XiTransform

    gamma_start = 1e-5
    gamma_max = 1e-1
    gamma_step = 10^(0.1)
    gamma = tf.Variable(gamma_start,dtype=tf.float64)
    gamma_incremented = tf.where(tf.less(gamma,gamma_max),gamma*gamma_step,gamma_max)
    op_increment_gamma = tf.assign(gamma,gamma_incremented)
    gamma_fallback = 1e-1
    op_gamma_fallback = tf.assign(gamma,gamma*gamma_fallback)
    sess = model[:enquire_session]()
    sess[:run](tf.variables_initializer([gamma]))
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
        op_adam = gpflow.train[:AdamOptimizer]()[:make_optimize_tensor](model)
    end

    for i in 1:iterations
        #
        # actions = [adam]
        try
            sess[:run](op_natgrad)
            sess[:run](op_increment_gamma)
        catch
            g = sess[:run](gamma)
            println("Gamma $g on iteration $i is too big: Falling back to $(g*gamma_fallback)")
            sess[:run](op_gamma_fallback)
        end
        if op_adam!=0
            sess[:run](op_adam)
        end
        if i % 100 == 0
            print("$i gamma=$(sess[:run](gamma)) ELBO=$(sess[:run](model[:likelihood_tensor]))")
        end
        if callback!= nothing
            callback(model,sess,i)
        end
    end
    model[:anchor](sess)
end
t_SV = @elapsed run_nat_grads_with_adam(model, iterations; ind_points_fixed=false, kernel_fixed=false,callback=pythonlogger)
y_SV = model[:predict_y](X_test)[1]
acc_SV = 1-sum(abs.(sign.(y_SV-0.5)-y_test))/(2*length(y_test))
println("SVGPC model : Acc=$(acc_SV), time=$t_SV")
ind_points = model[:feature][:Z][:value]
using Plots
pyplot()
p1=plot(x_test,x_test,reshape(y_test,N_test,N_test),t=:contour,cbar=false,fill=:true)
plot!(X[y.==1,1],X[y.==1,2],color=:red,t=:scatter,lab="y=1",title="Truth",xlims=(-5,5),ylims=(-5,5))
plot!(X[y.==-1,1],X[y.==-1,2],color=:blue,t=:scatter,lab="y=-1")
p2=plot(x_test,x_test,reshape(y_SV,N_test,N_test),t=:contour,fill=true,cbar=false,clims=(0,1),lab="",title="SVGPC")
plot!(ind_points[:,1],ind_points[:,2],t=:scatter,lab="Inducing Points")
display(plot(p1,p2));
