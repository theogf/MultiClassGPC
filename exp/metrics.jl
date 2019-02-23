function calibration(y_test,y_pred;nBins::Int=15,plothist=false,plotline=false,gpflow=false)
    edges = collect(range(0.0,1.0,length=nBins+1))
    mean_bins = 0.5*(edges[2:end]+edges[1:end-1])
    K = length(unique(y_test))
    nP = [zeros(nBins) for _ in 1:K]
    non_empty = falses(nBins)
    accs = [zeros(nBins) for _ in 1:K]
    conf = [zeros(nBins) for _ in 1:K]
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
    for k in unique(y_test)
        p=0
        for i in 1:length(y_test)
            if gpflow
                p = y_pred[k+bias,i]
            else
                p = y_pred[Symbol(k)][i]
            end
            bin = findlast(x->p>x,edges)
            nP[k][bin] += 1
            accs[k][bin] += k==y_test[i]
            conf[k][bin] += p
            non_empty[bin] = true
        end
        accs[k] ./= ifelse.(nP[k].!=0,nP[k],ones(length(nP[k])))
        conf[k] ./= ifelse.(nP[k].!=0,nP[k],ones(length(nP[k])))
        ECE[k] = sum(nP[k].*abs.(accs[k].-conf[k]))./length(y_test)
        MCE[k] = maximum(abs.(accs[k].-conf[k]))
    end

    ps = []
    if plotline
        for k in 1:K
            push!(ps,plot!(plot(conf[k][nP[k].!=0],accs[k][nP[k].!=0],title="y=$k",lab="",color=col_doc[k]),x->x,0:1,color=:black,lab="",xlabel="Confidence",ylabel="Accuracy",xlims=(0,1),ylims=(0,1)))
        end
        push!(ps,plot!(plot((sum((nP[k].!=0).*conf[k] for k in 1:K)./sum(nP[k].!=0 for k in 1:K))[non_empty],(sum(non_empty[k].*accs[k] for k in 1:K)./sum(nP[k].!=0 for k in 1:K))[non_empty],title="Mean",lab="",xlabel="Confidence",ylabel="Accuracy"),x->x,0:1,color=:black,lab="",xlims=(0,1),ylims=(0,1)))
        display(plot(ps...))
    end
    hists = []
    if plothist
        for k in 1:K
            push!(hists,bar(mean_bins,accs[k],title="y=$k",lab="",color=col_doc[k],xlims=(0,1),ylims=(0,1)))
        end
        push!(hists,bar(mean_bins,(sum(non_empty[k].*accs[k] for k in 1:K)./sum(nP[k].!=0 for k in 1:K))[non_empty],title="Mean",lab="",xlims=(0,1),ylims=(0,1)))
        display(plot(hists...))
    end
    if plothist && !plotline
        return ECE,MCE,plot(hists...)
    elseif plothist && plotline
        return ECE,MCE,plot(ps...),plot(hists...)
    elseif !plothist && plotline
        return ECE,MCE,plot(ps...)
    else
        return ECE,MCE
    end
end
