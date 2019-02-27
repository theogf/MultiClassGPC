using RCall
R"library(CalibratR)"

function calibration(y_test,y_pred;nBins::Int=10,plothist=false,plotline=false,gpflow=false)
    edges = collect(range(0.0,1.0,length=nBins+1))
    mean_bins = 0.5*(edges[2:end]+edges[1:end-1])
    ntest = length(y_test)
    K = length(unique(y_test))
    global nP = [zeros(Int64,nBins) for _ in 1:K]
    non_empty = falses(nBins)
    global accs = [zeros(nBins) for _ in 1:K]
    global conf = [zeros(nBins) for _ in 1:K]
    ECE = zeros(K)
    MCE = zeros(K)
    col_doc = []
    bias = 0
    if count(y_test.==0) != 0
        bias = 1
    end
    if K == 3
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
            bin = min(findlast(x->p>x,edges),nBins)
            nP[k][bin] += 1
            accs[k][bin] += k==(y_test[i]+bias)
            conf[k][bin] += p
            non_empty[bin] = true
        end
        accs[k] ./= ifelse.(nP[k].!=0,nP[k],ones(length(nP[k])))
        conf[k] ./= ifelse.(nP[k].!=0,nP[k],ones(length(nP[k])))
        ECE[k] = sum(nP[k].*abs.(accs[k].-conf[k]))./ntest
        MCE[k] = maximum(abs.(accs[k].-conf[k]))
    end

    ps = []
    msize = 20.0
    if plotline
        for k in 1:K
            push!(ps,plot!(plot(conf[k][nP[k].!=0],accs[k][nP[k].!=0],title="y=$(k-bias)",lab="",color=K==3 ? col_doc[k] : k,marker=:circle,markersize=msize*nP[k][nP[k].!=0]/ntest),x->x,0:1,color=:black,lab="",xlabel="Confidence",ylabel="Accuracy",xlims=(-0.1,1.1),ylims=(0,1)))
        end
        push!(ps,plot!(plot((sum(nP[k].*conf[k] for k in 1:K)./sum(nP[k] for k in 1:K))[non_empty],(sum(nP[k].*accs[k] for k in 1:K)./sum(nP[k] for k in 1:K))[non_empty],title="Mean",lab="",xlabel="Confidence",ylabel="Accuracy",marker=:circle,markersize=msize.*sum(nP)[sum(nP).!=0]./(3*ntest)),x->x,0:1,color=:black,lab="",xlims=(-0.1,1.1),ylims=(0,1)))
        display(plot(ps...))
    end
    hists = []
    if plothist
        for k in 1:K
            push!(hists,bar(mean_bins,accs[k],title="y=$k",lab="",color=K<=3 ? col_doc[k] : k,xlims=(0,1),ylims=(0,1)))
        end
        push!(hists,bar(mean_bins[non_empty],(sum(non_empty[k].*accs[k] for k in 1:K)./sum(nP[k].!=0 for k in 1:K))[non_empty],title="Mean",lab="",xlims=(0,1),ylims=(0,1)))
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
