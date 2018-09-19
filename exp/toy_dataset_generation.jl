using Distributions
using Plots
pyplot()
N = 1000
D = 10
y = zeros(Float64,N)
X = zeros(N,D)
for i in 1:N
    y[i] = sample(1:3,1)[1]
    if y[i] == 1
        r = rand(Uniform(0.1,0.5))
        θ = rand(Uniform(0,2π))
        X[i,1:2] = [r*cos(θ),r*sin(θ)]
    elseif y[i] == 2
        r = rand(Uniform(0.6,1.0))
        θ = rand(Uniform(0,2π))
        X[i,1:2] = [r*cos(θ),r*sin(θ)]
    elseif y[i] == 3
        X[i,1:2] = rand(Normal(0,0.01),2)
    end
    X[i,3:end] = rand(Normal(),D-2)
end

p = plot()
[plot!(p,X[y.==i,1],X[y.==i,2],t=:scatter) for i in 1:3]
plot(p)
gui()
