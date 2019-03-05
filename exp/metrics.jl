using RCall
using DelimitedFiles
using Plots
using Formatting
R"library(CalibratR)"

dpi=600
function calibration(y_test,y_pred;nBins::Int=10,plothist=false,plotline=false,gpflow=false,meanonly=false,threshold=0,title="")
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
    msize = 20/3*K
    if plotline
        if !meanonly
            for k in 1:K
                push!(ps,plot!(plot(conf[k][nP[k].!=0],accs[k][nP[k].!=0],title="y=$(k-bias)",lab="",color=K==3 ? col_doc[k] : k,marker=:circle,markersize=msize*nP[k][nP[k].!=0]/ntest),x->x,0:1,color=:black,lab="",xlabel="Confidence",ylabel="Accuracy",xlims=(-0.1,1.1),ylims=(0,1)))
            end
        end
        msizechoice = msize*tot_bin[tot_bin.>threshold]/(K*ntest)
        # push!(ps,plot!(plot((sum(nP[k].*conf[k] for k in 1:K)./sum(nP[k] for k in 1:K))[tot_bin.>threshold],(sum(nP[k].*accs[k] for k in 1:K)./sum(nP[k] for k in 1:K))[tot_bin.>threshold],lab="",xlabel="Confidence",ylabel="Accuracy",color=:black,marker=:circle,series_annotations=percent),x->x,0:1,color=:black,lab="",xlims=(-0.05,1.05),ylims=(0,1),dpi=dpi))
        push!(ps,plot!(plot((sum(nP[k].*conf[k] for k in 1:K)./sum(nP[k] for k in 1:K))[tot_bin.>threshold],(sum(nP[k].*accs[k] for k in 1:K)./sum(nP[k] for k in 1:K))[tot_bin.>threshold],lab="",xlabel="Confidence",ylabel="Accuracy",color=:black,marker=:circle,markersize=msizechoice),x->x,0:1,color=:black,lab="",xlims=(-0.05,1.05),ylims=(0,1),dpi=dpi))
        # display(plot(ps...))
    end
    hists = []
    if plothist
        if !meanonly
            for k in 1:K
                push!(hists,bar(mean_bins,accs[k],title="y=$k",lab="",color=K<=3 ? col_doc[k] : k,xlims=(0,1),ylims=(0,1),xlabel="Confidence",ylabel="Accuracy"))
            end
        end
        # g=:speed
        # color_grad = min.(tot_bin[tot_bin.>threshold]/(sum(tot_bin)),maximum(tot_bin[2:end]))
        # C(g) = RGB[g[z] for z=color_grad]
        # colors_grad = cgrad(g) |> C
        global percent = text.(string.(format.(tot_bin/(sum(tot_bin))*100,width=1,precision=1,suffix="%")),:bottom,12)
        push!(hists,bar(mean_bins[tot_bin.>threshold],(sum(accs[k][tot_bin.>threshold] for k in 1:K)./sum(nP[k].!=0 for k in 1:K)[tot_bin.>threshold]) ,lab="",xlims=(0,1),ylims=(0,1),bar_width=0.1,xlabel="Confidence",ylabel="Accuracy",dpi=dpi))
        # annotate!(hists[1],collect(zip(mean_bins[tot_bin.>threshold],sum(accs[k][tot_bin.>threshold] for k in 1:K)./sum(nP[k].!=0 for k in 1:K)[tot_bin.>threshold],percent)))
        # display(plot(hists...))
    end
    if plothist && !plotline
        return ECE,MCE,plot(hists...,title=title)
    elseif plothist && plotline
        return ECE,MCE,plot(ps...,title=title),plot(hists...,title=title)
    elseif !plothist && plotline
        return ECE,MCE,plot(ps...,title=title)
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
    σs = collect(0.1:0.1:0.7)
    nσ = length(σs)
    metrics = ["acc","ll"]
    labels = Dict("acc"=>"Test Error","ll"=>"Neg. Log Likelihood","ece"=>"Expected Calibration Error")
    defdict = Dict("acc"=>Float64[],"ll"=>Float64[],"ece"=>Float64[])
    global res = [("alsm",deepcopy(defdict)),("lsm",deepcopy(defdict)),("sm",deepcopy(defdict)),("rm",deepcopy(defdict)),("ep",deepcopy(defdict))]
    for σ in σs
        global vals = readdlm("resultslikelihood/results_$σ.txt")
        for i in eachindex(res)
            @show σ i
            push!(res[i][2]["acc"],1.0.-vals[1,i])
            push!(res[i][2]["ll"],-vals[2,i])
            push!(res[i][2]["ece"],vals[3,i])
        end
    end
    ps = []
    alllims1 = []
    alllims2 = []
    linewidth=3.0
    for (i,obj) in enumerate(res)
        p = plot(σs,obj[2][metrics[1]],xlabel="σ²",ylabel=labels[metrics[1]],lab="",linewidth=linewidth,tickfontcolor=:blue,color=:blue,title=obj[1])
        push!(alllims1,ylims(p))
        ptwin = twinx(p)
        plot!(ptwin,σs,obj[2][metrics[2]],xlabel="σ²",ylabel=labels[metrics[2]],lab="",linewidth=linewidth,tickfontcolor=:red,color=:red)
        push!(alllims2,ylims(ptwin))
        push!(ps,p)
    end
    minlim1 = Inf; maxlim1 = -Inf
    for (x,y) in alllims1
        minlim1 = min(x,minlim1); maxlim1 = max(y,maxlim1)
    end
    global llim1= (minlim1,maxlim1)
    minlim2 = Inf; maxlim2 = -Inf
    for (x,y) in alllims2
        minlim2 = min(x,minlim2); maxlim2 = max(y,maxlim2)
    end
    global lim2= (minlim2,maxlim2)
    ps=[]
    for (i,obj) in enumerate(res)
        p = plot(σs,obj[2][metrics[1]],xlabel="σ²",ylabel=labels[metrics[1]],lab="",linewidth=linewidth,guidefontcolor=:blue,ytickfontcolor=:blue,color=:blue,ylims=lim1)
        ptwin = twinx(p)
        plot!(ptwin,σs,obj[2][metrics[2]],xlabel="σ²",ylabel=labels[metrics[2]],lab="",linewidth=linewidth,yguidefontcolor=:red,ytickfontcolor=:red,color=:red,ylims=lim2)
        savefig(p,"../plotslikelihood/sigma_comparison_"*obj[1]*".pdf")
        title!(p,obj[1])
        push!(ps,p)
    end

    plot(ps...,link=:all,layout=(1,length(res)))
end

function plot_likelihood_diff2()
    σs = collect(0.1:0.1:0.7)
    nσ = length(σs)
    metrics = ["acc","ll","ece"]
    global labels = Dict("acc"=>"Test Error","ll"=>"Neg. Log Likelihood","ece"=>"Expected Calibration Error","sm"=>"Softmax","lsm"=>"Logistic Softmax","rm"=>"Robust-max","ep"=>"Multinomial probit")
    defdict = Dict("acc"=>Float64[],"ll"=>Float64[],"ece"=>Float64[])
    global res = [("lsm",deepcopy(defdict)),("sm",deepcopy(defdict)),("rm",deepcopy(defdict)),("ep",deepcopy(defdict))]
    for σ in σs
        global vals = readdlm("resultslikelihood/results_$σ.txt")
        for i in 1:length(res)
            @show σ i
            push!(res[i][2]["acc"],1.0.-vals[1,i+1])
            push!(res[i][2]["ll"],-vals[2,i+1])
            push!(res[i][2]["ece"],vals[3,i+1])
        end
    end
    ps = []
    for m in metrics
        p = plot()
        for i in eachindex(res)
            plot!(p,σs,res[i][2][m],xlabel="σ²",ylabel=labels[m],lab="",linewidth=2.0,color=i,xlims=(0.1,maximum(σs)))
        end
        savefig(p,"../plotslikelihood/"*m*"comparison.pdf")
        push!(ps,p)
    end
    p = plot(ones(1,length(res)),ones(1,length(res)),linewidth=2.0,label=hcat(get.([labels],getindex.(res,[1]),nothing)...), grid=false, showaxis=false,legend=:left,legendfontsize=20.0,color=collect(1:length(res))')
    savefig(p,"../plotslikelihood/legendcomparison.pdf")
    push!(ps,p)
    plot(ps...,link=:all,layout=(1,length(metrics)+1))
end

function calibration_plots(dataset::String,write=false)
    cd(@__DIR__)
    methods = ["SCGPMC","SCGPMC_shared","SVGPMC","EPGPMC"]
    global results = Dict{String,Matrix}()
    plotshist = Dict{String,Any}()
    plotsline = Dict{String,Any}()
    y_test = []
    for m in methods
        data = readdlm("../cluster/AT_S_Experiment/"*dataset*"Dataset/y_prob_"*m*".txt")
        results[m] = data[:,2:end]
        y_test = Int64.(data[:,1])
        n = length(y_test)
        ECE,MCE,plotsline[m],plotshist[m] = calibration(y_test,results[m],plothist=true,plotline=true,meanonly=true,gpflow=true,title=m,threshold=0)#div(n,100))
    end
    display(plot(values(plotsline)...))
    savefig(plot(values(plotsline)...),"resultsexps/"*dataset*"_linecalibration.png")
    display(plot(values(plotshist)...))
    savefig(plot(values(plotshist)...),"resultsexps/"*dataset*"_histcalibration.png")
end
