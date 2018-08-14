using Distributions
using PyCall
using Plots
@pyimport gpflow
@pyimport tensorflow as tf
using OMGP

N_data = 100000
N_test = 40
N_indpoints = 20
N_dim = 2
noise = 0.2
minx=-5.0
maxx=5.0
function latent(x)
    return x[:,1].*sin.(0.5*x[:,2])
end

X = rand(N_data,N_dim)*(maxx-minx)+minx
x_test = linspace(minx,maxx,N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = sign.(latent(X)+rand(Normal(0,noise),size(X,1)))
y_test = sign.(latent(X_test)+rand(Normal(0,noise),size(X_test,1)))
(nSamples,nFeatures) = (N_data,N_dim)
m=20
batchsize=40
iterations=10000
Z = KMeansInducingPoints(X,m,10)
l=1.0
kernel = gpflow.kernels[:RBF](N_dim,lengthscales=l,ARD=false)+gpflow.kernels[:White](N_dim,variance=noise)
model = gpflow.models[:SVGP](X, reshape((y+1)./2,(length(y),1)),num_latent=1,kern=kernel, likelihood=gpflow.likelihoods[:Bernoulli](), Z=Z, minibatch_size=batchsize)
LogArrays = Array{Any,1}()
iter_points = vcat(1:99,100:10:999,1000:100:9999)
function TestAccuracy(y_test, y_predic)
  return 1-sum(1-y_test.*y_predic)/(2*length(y_test))
end
function pythonlogger(model,session,iter)
      if in(iter,iter_points)
          a = zeros(8)
          a[1] = time_ns()
          y_p = model[:predict_y](X_test)[1]
          loglike = zeros(y_p)
          loglike[y_test.==1] = log.(y_p[y_test.==1])
          loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
          a[2] = TestAccuracy(y_test,sign.(y_p-0.5))
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
t_SV = @elapsed run_nat_grads_with_adam(model, iterations; ind_points_fixed=false, kernel_fixed =false,callback=pythonlogger)
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
