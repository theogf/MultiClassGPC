using RCall
R"library(CalibratR)"

dpi=600
function calibration(y_test,y_pred;nBins::Int=10,plothist=false,plotline=false,gpflow=false,meanonly=false,threshold=0)
    edges = collect(range(0.0,1.0,length=nBins+1))
    mean_bins = 0.5*(edges[2:end]+edges[1:end-1])
    ntest = length(y_test)
    K = length(unique(y_test))
    global nP = [zeros(Int64,nBins) for _ in 1:K]
    global tot_bin = zeros(Int,nBins)
    global accs = [zeros(nBins) for _ in 1:K]
    global conf = [zeros(nBins) for _ in 1:K]
    ECE = zeros(Float64,K)
    MCE = zeros(Float64,K)
    col_doc = []
    bias = 0
    if count(y_test.==0) != 0
        bias = 1
    end
    if K == 3 && (plothist || plotline)
        col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    end
    for k in unique(y_test).+bias
        p=0
        for i in 1:ntest
            if gpflow
                p = y_pred[i,k]
            else
                p = y_pred[Symbol(k-bias)][i]
            end
            p = min(max(0.0,p),1.0)
            bin = min(findlast(x->p>=x,edges),nBins)
            nP[Int(k)][bin] += 1
            accs[Int(k)][bin] += k==(y_test[i]+bias)
            conf[Int(k)][bin] += p
            tot_bin[bin] += 1
        end
        accs[Int(k)] ./= ifelse.(nP[Int(k)].!=0,nP[Int(k)],ones(length(nP[Int(k)])))
        conf[Int(k)] ./= ifelse.(nP[Int(k)].!=0,nP[Int(k)],ones(length(nP[Int(k)])))
        ECE[Int(k)] = sum(nP[Int(k)].*abs.(accs[Int(k)].-conf[Int(k)]))./ntest
        MCE[Int(k)] = maximum(abs.(accs[Int(k)].-conf[Int(k)]))
    end

    ps = []
    msize = 20.0
    if plotline
        if !meanonly
            for k in 1:K
                push!(ps,plot!(plot(conf[k][nP[k].!=0],accs[k][nP[k].!=0],title="y=$(k-bias)",lab="",color=K==3 ? col_doc[k] : k,marker=:circle,markersize=msize*nP[k][nP[k].!=0]/ntest),x->x,0:1,color=:black,lab="",xlabel="Confidence",ylabel="Accuracy",xlims=(-0.1,1.1),ylims=(0,1)))
            end
        end
        push!(ps,plot!(plot((sum(nP[k].*conf[k] for k in 1:K)./sum(nP[k] for k in 1:K))[tot_bin.>threshold],(sum(nP[k].*accs[k] for k in 1:K)./sum(nP[k] for k in 1:K))[tot_bin.>threshold],lab="",xlabel="Confidence",ylabel="Accuracy",color=:black,marker=:circle,markersize=(msize.*tot_bin/(3*ntest))[tot_bin.>threshold]),x->x,0:1,color=:black,lab="",xlims=(-0.05,1.05),ylims=(0,1),dpi=dpi))
        display(plot(ps...))
    end
    hists = []
    if plothist
        if !meanonly
            for k in 1:K
                push!(hists,bar(mean_bins,accs[k],title="y=$k",lab="",color=K<=3 ? col_doc[k] : k,xlims=(0,1),ylims=(0,1),xlabel="Confidence",ylabel="Accuracy"))
            end
        end
        push!(hists,bar(mean_bins[tot_bin.>threshold],(sum(accs[k][tot_bin.>threshold] for k in 1:K)./sum(nP[k].!=0 for k in 1:K)[tot_bin.>threshold]) ,lab="",xlims=(0,1),ylims=(0,1),bar_width=0.1,xlabel="Confidence",ylabel="Accuracy",dpi=dpi))
        display(plot(hists...))
    end
    if plothist && !plotline
        return ECE,MCE,plot(hists...)
    elseif plothist && plotline
        return ECE,MCE,plot(ps...),plot(hists...),nP,accs,conf
    elseif !plothist && plotline
        return ECE,MCE,plot(ps...)
    else
        return ECE,MCE
    end
end


function calibration_R(y_test,y_pred;nBins::Int=15,gpflow=false)
    K = length(unique(y_test))
    bias = 0
    if count(y_test.==0) != 0
        bias = 1
    end
    ECE = zeros(K)
    MCE = zeros(K)
    for k in unique(y_test).+bias
        if gpflow
            p = y_pred[:,k]
        else
            p = vec(y_pred[Symbol(k-bias)])
        end
        y = Int64.(y_test.==(k-bias))
        ECE[k] = rcopy(R"CalibratR:::get_ECE_equal_width($y,$p,bins=10)")
        MCE[k] = rcopy(R"CalibratR:::get_MCE_equal_width($y,$p,bins=10)")
    end
    return ECE,MCE
end

function plot_likelihood_diff()
    σs = collect(0.1:0.1:0.6)
    nσ = length(σs)
    defdict = Dict("acc"=>Float64[],"ll"=>Float64[],"ece"=>Float64[])
    global res = [("alsm",deepcopy(defdict)),("lsm",deepcopy(defdict)),("sm",deepcopy(defdict)),("rm",deepcopy(defdict))]
    for σ in σs
        global vals = readdlm("resultslikelihood/results_$σ.txt")
        for i in eachindex(res)
            # @show obj
            push!(res[i][2]["acc"],vals[1,i])
            push!(res[i][2]["ll"],vals[2,i])
            push!(res[i][2]["ece"],vals[3,i])
        end
    end
    ps = []
    allacclims = []
    alllllims = []
    linewidth=3.0
    for (i,obj) in enumerate(res)
        p = plot(σs,1.0.-obj[2]["acc"],xlabel="σ²",ylabel="Test Error",lab="",linewidth=linewidth,tickfontcolor=:blue,color=:blue)
        push!(allacclims,ylims(p))
        ptwin = twinx(p)
        plot!(ptwin,σs,-obj[2]["ll"],xlabel="σ²",ylabel="Neg. Log Likelihood",lab="",linewidth=linewidth,tickfontcolor=:red,color=:red)
        push!(alllllims,ylims(ptwin))
        push!(ps,p)
    end
    minlimacc = Inf; maxlimacc = -Inf
    for (x,y) in allacclims
        minlimacc = min(x,minlimacc); maxlimacc = max(y,maxlimacc)
    end
    limacc= (minlimacc,maxlimacc)
    minlimll = Inf; maxlimll = -Inf
    for (x,y) in alllllims
        minlimll = min(x,minlimll); maxlimll = max(y,maxlimll)
    end
    limll= (minlimll,maxlimll)
    ps=[]
    for (i,obj) in enumerate(res)
        p = plot(σs,1.0.-obj[2]["acc"],xlabel="σ²",ylabel="Test Error",lab="",linewidth=linewidth,guidefontcolor=:blue,ytickfontcolor=:blue,color=:blue,ylims=limacc)
        push!(allacclims,ylims(p))
        ptwin = twinx(p)
        plot!(ptwin,σs,-obj[2]["ll"],xlabel="σ²",ylabel="Neg. Log Likelihood",lab="",linewidth=linewidth,yguidefontcolor=:red,ytickfontcolor=:red,color=:red,ylims=limll)
        push!(alllllims,ylims(ptwin))
        savefig(p,"../plotslikelihood/sigma_comparison_"*obj[1]*".pdf")
        push!(ps,p)
    end

    plot(ps...,link=:all,layout=(1,length(res)))
end
