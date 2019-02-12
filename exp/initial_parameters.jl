
#Create a default dictionary
function DefaultParameters()
  param = Dict{String,Any}()
  param["ϵ"]= 1e-8 #Convergence criteria
  param["BatchSize"] = 10 #Number of points used for stochasticity
  param["Kernel"] = "rbf" # Kernel function
  param["Θ"] = 1.0 # Hyperparameter for the kernel function
  param["γ"] = 1e-3 #Variance of introduced noise
  param["M"] = 32 #Number of inducing points
  param["Window"] = 5 #Number of points used to check convergence (smoothing effect)
  param["Verbose"] = 0 #Verbose
  param["Autotuning"] = false
  param["ConvCriter"] = "HOML"
  param["PointOptimization"] = false
  param["FixedInitialization"] = true
  param["nClasses"] = 0
  return param
end

#Create a default parameters dictionary for CGPC
function CGPMCParameters(;Stochastic=true,Sparse=true,ALR=true,Autotuning=false,independent=true,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["nSamples"] = main_param["nSamples"]
  param["Stochastic"] = Stochastic #Is the method stochastic
  param["Sparse"] = Sparse #Is the method using inducing points
  param["ALR"] = ALR #Is the method using adpative learning rate (in case of the stochastic case)
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ATFrequency"] = param["Stochastic"] ? 3 : 1 #Number of iterations between every autotuning
  param["κ_s"] = 1.0;  param["τ_s"] = 40; #Parameters for learning rate of Stochastic gradient descent when ALR is not used
  param["ϵ"] = main_param["ϵ"]; param["Window"] = main_param["Window"]; #Convergence criteria (checking parameters norm variation on a window)
  param["ConvCriter"] = main_param["ConvCriter"]
  if main_param["Kernel"] == "rbf"
    param["Kernel"] = AugmentedGaussianProcesses.RBFKernel(main_param["Θ"],variance=main_param["var"]) #Kernel creation (standardized for now)
  else
    param["Kernel"] = AugmentedGaussianProcesses.RBFKernel([main_param["Θ"]],dim=main_param["nFeatures"],variance=main_param["var"]) #Kernel creation (standardized for now)
  end
  param["Verbose"] = if typeof(main_param["Verbose"]) == Bool; main_param["Verbose"] ? 2 : 0; else; param["Verbose"] = main_param["Verbose"]; end; #Verbose
  param["BatchSize"] = main_param["BatchSize"] #Number of points used for stochasticity
  param["FixedInitialization"] = main_param["FixedInitialization"]
  param["M"] = main_param["M"] #Number of inducing points
  param["γ"] = main_param["γ"] #Variance of introduced noise
  param["independent"] = independent #Are GPs independent
  return param
end

#Create a default parameters dictionary for SVGPMC (similar to CGPMC)
function SVGPMCParameters(;Stochastic=true,main_param=DefaultParameters(),dohybrid=false)
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["nSamples"] = main_param["nSamples"]
  param["Sparse"] = true
  param["Stochastic"] = Stochastic
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ϵ"] = main_param["ϵ"]
  if main_param["Kernel"] == "rbf"
  param["Kernel"] =   gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=false)
  else
    param["Kernel"] = gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=true)
  end
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  param["SmoothingWindow"] = main_param["Window"]
  param["ConvCriter"] = main_param["ConvCriter"]
  param["nConjugateSteps"] = 200
  return param
end

#Create a default parameters dictionary for EPGPC (similar to CGPMC)
function EPGPMCParameters(;Stochastic=true,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["nSamples"] = main_param["nSamples"]
  param["Stochastic"] = Stochastic
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ϵ"] = main_param["ϵ"]
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  param["SmoothingWindow"] = main_param["Window"]
  param["ConvCriter"] = main_param["ConvCriter"]
  return param
end

#Create a default parameters dictionary for TTGPC (similar to CGPMC)
function TTGPMCParameters(;Stochastic=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ϵ"] = main_param["ϵ"]
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  param["SmoothingWindow"] = main_param["Window"]
  param["ConvCriter"] = main_param["ConvCriter"]
  return param
end


#Create a default parameters dictionary for ARMC (similar to CGPMC)
function ARMCParameters(;Stochastic=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["nClasses"] = main_param["nClasses"]
  param["maxIter"]=main_param["maxIter"]
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  param["SmoothingWindow"] = main_param["Window"]
  param["ConvCriter"] = main_param["ConvCriter"]
  return param
end
