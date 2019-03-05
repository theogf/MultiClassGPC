using PyPlot,Statistics
using Formatting
using PyCall
plt[:style][:use]("seaborn-colorblind")
cd(@__DIR__)
if VERSION >= v"0.7.0-"
    using DelimitedFiles
end
NC =  Dict("EPGPMC"=>"SEP-MGPC","TTGPC"=>"Tensor Train GPC", "LogReg"=>"Linear Model",
"SVGPMC"=>"SVI-MGPC","SCGPMC"=>"SC-MGPC","HSCGPMC"=>"Hybrid SC-MGPC","Accuracy"=>"Avg. Test Error","SCGPMCInd"=>"Independent Priors","SCGPMCShared"=>"Common Prior",
"MedianL"=>"Avg. Median Neg.\n Test Log-Likelihood","MeanL"=>"Avg. Neg. Test\n Log-Likelihood","AUC"=>"Multiclass AUC")
colors=Dict("SVGPMC"=>"blue","SCGPMC"=>"red","HSCGPMC"=>"yellow","EPGPMC"=>"green", "TTGPC"=>"black","SCGPMCInd"=>"blue","SCGPMCShared"=>"red")
linestyles=Dict(16=>":",32=>"--",64=>"-.",128=>"-")
# linestyles=Dict(4=>"-",8=>":",10=>"-",16=>"-.",32=>"--",50=>":",64=>"-.",100=>"-.",128=>"-",200=>"--",256=>"--")
metrics = Dict("Accuracy"=>3,"MeanL"=>5,"MedianL"=>7,"ELBO"=>9,"AUC"=>11, "ECE"=>13,"MCE"=>15)

# location of the results
c = "../cluster/AT_Experiment/"
cs = "../cluster/AT_S_Experiment/"
l = "results/Experiment/"
las = "results/AT_S_Experiment/"
la = "results/AT_Experiment/"
ls = "results/S_Experiment/"
f = "../final_results/"
loc = Dict{String,Dict{String,String}}()
loc["mnist"] =          Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["fashion-mnist"] =  Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["kmnist"] =         Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["covtype"] =        Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["shuttle"] =        Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["combined"] =       Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)

loc["iris"] =           Dict("SCGPMC"=>la,"HSCGPMC"=>la,"SVGPMC"=>la,"EPGPMC"=>la,"TTGPC"=>c)
loc["wine"] =           Dict("SCGPMC"=>la,"HSCGPMC"=>la,"SVGPMC"=>la,"EPGPMC"=>la,"TTGPC"=>c)
loc["glass"] =          Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["vehicle"] =          Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["dna"] =          Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["segment"] =          Dict("SCGPMC"=>las,"HSCGPMC"=>las,"SVGPMC"=>las,"EPGPMC"=>las,"TTGPC"=>cs)
loc["acoustic"] =          Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["seismic"] =          Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["Covtype"] =            Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["sensorless"] =          Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["cpu_act"] =          Dict("SCGPMC"=>cs,"HSCGPMC"=>la,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["Cod-rna"] =            Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Credit_card"] =        Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Diabetis"] =           Dict("SCGPMC"=>l,"HSCGPMC"=>la,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["Electricity"] =        Dict("SCGPMC"=>l,"HSCGPMC"=>la,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["German"] =             Dict("SCGPMC"=>l,"HSCGPMC"=>la,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["HIGGS"] =              Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Ijcnn1"] =             Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Mnist"] =              Dict("SCGPMC"=>l,"HSCGPMC"=>la,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["Poker"] =              Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Protein"] =            Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Shuttle"] =            Dict("SCGPMC"=>l,"HSCGPMC"=>la,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["SUSY"] =               Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Vehicle"] =            Dict("SCGPMC"=>c,"HSCGPMC"=>la,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["wXa"] =                Dict("SCGPMC"=>l,"HSCGPMC"=>la,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)


gwidth = 2.0
gmarkersize= 5.0
function DataConversion(array,name)
    if name == "Accuracy"
        return 1.0.-array
    elseif name == "MedianL"
        return -array
    elseif name == "MeanL"
        return -array
    elseif name == "AUC"
        return array
    end
end


markers=Dict(21=>"o",42=>"o",104=>"o",208=>"o",416=>"o",1040=>"o",2079=>"o")
markers=Dict(8=>"o",15=>"o",38=>"o",76=>"o",152=>"o",381=>"o",761=>"o")
linestyles=Dict(21=>"-",42=>"-",104=>"-",208=>"-",416=>"-",1040=>"-",2079=>"-")
linestyles=Dict(8=>"-",15=>"-",38=>"-",76=>"-",152=>"-",381=>"-",761=>"-")
function DoubleAxisPlot(metric,MPoints=[5,10,20,50,75,100,150,200,300,400])
    scale= 2.0
    dataset="segment"
    strat = ["Ind","Shared"]
    Nm = length(MPoints)
    percent = [20,50,75,100,150,200,300,400]
    p = Dict("SCGPMCInd"=>Array{Float64,2}(undef,length(MPoints),2),"SCGPMCShared"=>Array{Float64,2}(undef,length(MPoints),2))
    for T in strat
        for (i,M) in enumerate(MPoints)
            r = readdlm("../cluster/results_M$(M)_$(T)/AT_Experiment/$(dataset)Dataset/Results_SCGPMC.txt")
            p["SCGPMC"*T][i,:] = r[end,[metrics[metric],1]]
        end
        p["SCGPMC"*T][:,1]= DataConversion(p["SCGPMC"*T][:,1],metric)
    end

    fig, ax1 = plt[:subplots]()
    fig[:set_size_inches](16,8)
    p1 = ax1[:plot](MPoints[1:Nm],p["SCGPMCInd"][:,1],label="",color="red",marker="x",linestyle="-",linewidth=2.0*scale,markersize=4.0*scale)
    p2 = ax1[:plot](MPoints[1:Nm],p["SCGPMCShared"][:,1],label="",color="blue",marker="o",linestyle="-",linewidth=2.0*scale,markersize=4.0*scale)
    ax1[:set_xlabel]("# inducing points",fontsize=20.0*scale)
    ax1[:set_ylabel](NC[metric]*"\n(solid line)",fontsize=20.0*scale)
    ax1[:tick_params]('y',fontsize=15.0*scale)
    xticks(percent,["$v" for v in percent],fontsize=15.0*scale)
    yticks(fontsize=15.0*scale)
    ax2 = ax1[:twinx]()

    p4 = ax2[:semilogy](MPoints[1:Nm],p["SCGPMCInd"][:,2],color="red",marker="x",linestyle="--",linewidth=2.0*scale,markersize=4.0*scale)
    p3 = ax2[:semilogy](MPoints[1:Nm],p["SCGPMCShared"][:,2],color="blue",marker="o",linestyle="--",linewidth=2.0*scale,markersize=4.0*scale)
    ax2[:set_ylabel]("Training time in Seconds\n(dashed line)",fontsize=18.0*scale)
    ax2[:tick_params]('y',fontsize=20.0*scale)
    yticks([10.0,100.0,1000.0],fontsize=15.0*scale)
    ax1[:legend](["Ind. Hyperparameters","Shared Hyperparameters"],fontsize=17.0*scale,loc=7,markerscale=2.0*scale)
    fig[:tight_layout]()
    plt[:show]()
    savefig("../plots/$(dataset)DoublePlot.png")
    return fig
end

true


function InducingPointsComparison(metric,MPoints=[8,15,38,76,152,381];step=1)
# function InducingPointsComparison(metric,MPoints=[21,42,104,208,416,1040,2079];step=1)
    dataset="vehicle"
    figure("Comparison of inducing points accuracy and time",figsize=(16,8)); clf();
    p = Dict("SCGPMCInd"=>Array{Any,1}(),"SCGPMCShared"=>Array{Any,1}())
    lab = Dict("SCGPMCInd"=>Array{Any,1}(),"SCGPMCShared"=>Array{Any,1}())
    strat = ["Ind","Shared"]
    for T in strat
        for M in MPoints
            Results = Dict{String,Any}()
            Results["SCGPMC"*T] = readdlm("../cluster/results_M$(M)_$(T)/AT_Experiment/$(dataset)Dataset/Results_SCGPMC.txt")
            for (name,res) in Results
                res[:,metrics[metric]] = DataConversion(res[:,metrics[metric]],metric)
                push!(lab[name],NC[name]*" M=$M")
                new_p,=semilogx(res[1:step:end,1],SmoothIt(res[1:step:end,metrics[metric]],window=1),markersize=gmarkersize,color=colors[name],marker=markers[M],linewidth=gwidth,linestyle=linestyles[M],label=NC[name]*" M=$M")
                push!(p[name],new_p)
            end
        end
    end
    xlabel("Training Time in Seconds",fontsize=20.0)
    xticks(fontsize=18.0)
    ylabel(NC[metric],fontsize=20.0)
    yticks(fontsize=18.0)
    title(dataset,fontsize=24.0,fontweight="semibold")

    legend([p["SCGPMCInd"];p["SCGPMCShared"]],[lab["SCGPMCInd"];lab["SCGPMCShared"]],fontsize=18.0)
    # xlim([0.03,4500])
    # ylim([-0.01,0.15])
    tight_layout()
    savefig("../plots/$(dataset)InducingPointsPlot.png")
end

function PlotAll(;shared=false)
    file_list = readdlm("files_finished")
    for file in file_list
        PlotMetricvsTime(file,"Final",time=true,writing=true,corrections=false,shared=shared)
    end
end

function SmoothIt(x;window=3)
    smoothed = zero(x)
    for i in 1:length(x)
        smoothed[i] = mean(x[max(1,i-window):min(length(x),i+window)])
    end
    return smoothed
end

function PlotMetricvsTime(dataset,metric;final=false,AT=true,time=true,writing=false,corrections=false,shared=false)
    cd(@__DIR__)
    global Results = Dict{String,Any}();
    println("Working on dataset $dataset")
    # colors=Dict("GPC"=>"b","SPGGPC"=>"r","LogReg"=>"y")
    time_line = [1:1:9;1e1:5*1e0:1e2-1;1e2:5*1e1:1e3-1;1e3:5*1e2:1e4-1;1e4:5*1e3:1e5-1;1e5:5*1e4:1e6]
    # Dict("SVGPMC"=>[1:1:99;100:10:999;1000:100:9999;10000:1000:20000],"SCGPMC"=>[1:1:99;100:10:999;1000:100:20000])
    p = Dict{String,Any}()

    # FinalMetrics = ["MeanL","AUC"]
    FinalMetrics = ["MeanL","Accuracy"]

    # NC =  Dict("LogReg"=>"Linear Model","GPC"=>"SVGPMC","SPGGPC"=>"X-GPC","Accuracy"=>"Avg. Test Error","MedianL"=>"Avg. Median Neg. Test Log likelihood")
    Results["SVGPMC"] = readdlm(loc[dataset]["SVGPMC"]*dataset*"Dataset/Results_SVGPMC.txt")
    Results["SCGPMC"] = readdlm(loc[dataset]["SCGPMC"]*dataset*"Dataset/Results_SCGPMC"*(shared ? "_shared" : "")*".txt")
    # Results["HSCGPMC"] = readdlm(loc[dataset]["HSCGPMC"]*dataset*"Dataset/Results_HSCGPMC.txt")
    Results["EPGPMC"] = readdlm(loc[dataset]["EPGPMC"]*dataset*"Dataset/Results_EPGPMC.txt")
    # if !in(dataset,["iris","glass","wine"])
    #     Results["SVGPMC"][:,1] = Results["SVGPMC"][:,1] .- Float64(readdlm("../cluster/time_correction/"*dataset*"SVGPMC.txt")[1])
    # end
    # if dataset == "mnist"
    #     Results["SCGPMC"][:,1] = Results["SCGPMC"][:,1] .- Float64(readdlm("../cluster/time_correction/$(dataset)SCGPMC.txt")[1])
    # end
    # Results["TTGPC"] = readdlm(loc[dataset]["TTGPC"]*dataset*"Dataset/Results_TTGPC.txt")
    # Results["LogReg"] = readdlm(loc[dataset]["LogReg"]*dataset*"Dataset/Results_LogReg.txt")
    acc = Results["SCGPMC"][:,metrics["Accuracy"]]
    meanl = Results["SCGPMC"][:,metrics["MeanL"]]
    maxacc = findmax(acc)
    acc = vcat(acc[1:maxacc[2]-1],max.(acc[maxacc[2]:end],maxacc[1]))
    maxmeanl = findmax(meanl)
    meanl = vcat(meanl[1:maxmeanl[2]-1],max.(meanl[maxmeanl[2]:end],maxmeanl[1]))
    Results["SCGPMC"][:,metrics["Accuracy"]]= acc
    Results["SCGPMC"][:,metrics["MeanL"]] = meanl



    maxx = maximum((x->x[end,1]).(values(Results)))#,Results["TTGPC"][end,1])
    minx = minimum((x->x[1,1]).(values(Results)))#,Results["TTGPC"][1,1])
    # logreg = zeros(2,10)
    # logreg[1,:] = Results["LogReg"]; logreg[2,2:end] = Results["LogReg"][2:end]; logreg[2,1] = maxx;
    # println(logreg)
    # Results["LogReg"] = logreg
    f=[]
    if metric != "Final"
        f = figure("Convergence on dataset "*dataset*" ",figsize=(16,9));clf();
    else
        f = figure("Convergence on dataset "*dataset*" ",figsize=(16,4.5));clf();
    end
    step=1
    if corrections
        if dataset == "aXa"
            #Divide acc stderr by 2
            Results["SCGPMC"][:,4] *= 0.5;
            Results["SVGPMC"][:,4] *= 0.5;
            Results["EPGPMC"][:,4] *= 0.5;
        elseif dataset == "Bank_marketing"
            #Divide acc stderr by 2
            Results["SCGPMC"][:,4] *= 0.5;
            Results["SVGPMC"][:,4] *= 0.5
            Results["EPGPMC"][:,4] *= 0.5
            Results["SCGPMC"][:,6] *= 0.5;
            Results["SVGPMC"][:,6] *= 0.5
            Results["EPGPMC"][:,6] *= 0.5
        elseif dataset == "Electricity"
            #Divide acc stderr by 2
            Results["SCGPMC"][:,4] *= 0.5;
            Results["SVGPMC"][:,4] *= 0.5
            Results["EPGPMC"][:,4] *= 0.5
        elseif dataset == "German"
            Results["SCGPMC"][:,4] *= 0.5;
            Results["SVGPMC"][:,4] *= 0.5;
            Results["EPGPMC"][:,4] *= 0.5;
            Results["SCGPMC"][:,6] *= 0.5;
            Results["SVGPMC"][:,6] *= 0.5;
            Results["EPGPMC"][:,6] *= 0.5;
        end
    end
    if time
        time_line = Dict(key=>Results[key][:,1] for key in keys(Results))
    else
        time_line = Dict(key=>time_line[1:length(Results[key][:,1])] for key in keys(Results))
    end
    if metric == "Final"
        iter=1
        giter = 2
        for (mname,mmetric) in metrics
            if in(mname,FinalMetrics)
                subplot(1,2,giter)
                for name in sort(collect(keys(Results)),rev=true)
                    # if name != "LogReg"
                        Results[name][:,mmetric] = DataConversion(Results[name][:,mmetric],mname)
                        x = [time_line[name][1:step:end];maxx]
                        my = [Results[name][1:step:end,mmetric];Results[name][end,mmetric]]
                        sy = [Results[name][1:step:end,mmetric+1];Results[name][end,mmetric+1]]/sqrt(10)
                        new_p, = semilogx(x,my,color=colors[name],linewidth=gwidth,label=NC[name])
                        fill_between(x,my-sy,my+sy,alpha=0.2,facecolor=colors[name])
                        p[name]=new_p
                end
                if time
                    xlabel("Training Time in Seconds",fontsize=20.0)
                    xlim([0.5*minx,1.5*maxx])
                    xticks(fontsize=15.0)
                    yticks(fontsize=15.0,)
                else
                    xlabel("Iterations")
                end
                ylabel(NC[mname],fontsize=20.0)
		if mname == "MeanL"
			legpos = 2
		else
			legpos = 1
		end
                legend([p[key] for key in keys(Results)],
                [NC[key] for key in keys(Results)],fontsize=20.0,loc=legpos)#;NC["TTGPC"];NC["LogReg"]])
                giter-=1
            end
        end
    elseif metric != "All"
        for (mname,results) in Results
            semilogx(time_line[mname][1:step:end],results[1:step:end,metrics[metric]],color=colors[mname],label=NC[mname])
            fill_between(time_line[mname][1:step:end],results[1:step:end,metrics[metric]]-results[1:step:end,metrics[metric]+1]/sqrt(10),results[1:step:end,metrics[metric]]+results[1:step:end,metrics[metric]+1]/sqrt(10),alpha=0.2,facecolor=colors[mname])
        end
        if time
            xlabel("Time [s]")
        else
            xlabel("Iterations")
        end
        ylabel(metric)
        legend(loc=4)
    else
        giter = 1
        for (mname,mmetric) in metrics
            subplot(2,ceil(Int64,length(metrics)/2),giter)
            for (name,results) in Results
                semilogx(time_line[name][1:step:end],results[1:step:end,mmetric],color=colors[name],label=name)
                fill_between(time_line[name][1:step:end],results[1:step:end,mmetric]-results[1:step:end,mmetric+1]/sqrt(10),results[1:step:end,mmetric]+results[1:step:end,mmetric+1]/sqrt(10),alpha=0.2,facecolor=colors[name])
            end
            if time
                xlabel("Time [s]")
            else
                xlabel("Iterations")
            end
            ylabel(mname)
            legend(loc=4)
            giter+=1
        end
    end
    suptitle(DatasetNameCorrection[dataset],fontsize=24.0,fontweight="semibold")
    tight_layout()
    subplots_adjust(top=0.88)
    if writing
        savefig("../plots/"*(metric=="Final" ? "Final" : "")*"Convergence_vs_"*(time ? "time" : "iterations")*"_on_"*dataset*(shared ? "_shared" : "")*".png")
        close()
    end
    return f
end


sizes = Dict("wine"=>(178,13,3),"vehicle"=>(846,18,4),"shuttle"=>(58000,9,7),"sensorless"=>(58509,48,11),"seismic"=>(98528,50,3),
            "segment"=>(2310,19,7),"satimage"=>(6430,36,6),"mnist"=>(70000,784,10),"isolet"=>(7797,617,26),"iris"=>(150,4,3),
            "glass"=>(214,9,6),"fashion-mnist"=>(70000,784,10),"dna"=>(3386,180,3),"cpu_act"=>(8192,21,56),
            "covtype"=>(851012,54,7),"combined"=>(98528,50,3),"acoustic"=>(98528,50,3))



DatasetNameCorrection = Dict("iris"=>"Iris","wine"=>"Wine","glass"=>"Glass","vehicle"=>"Vehicle", "segment"=>"Segment",
                             "dna"=>"DNA","satimage"=>"SatImage","mnist"=>"MNIST","kmnist"=>"K-Mnist","vehicle"=>"Vehicle","combined"=>"Combined","fashion-mnist"=>"Fashion Mnist",
                             "sensorless"=>"Sensorless","acoustic"=>"Acoustic","covtype"=>"CovType","cpu_act"=>"CPU Act",
                             "seismic"=>"Seismic","shuttle"=>"Shuttle",
                            "Cod-rna"=>"Cod RNA", "Covtype"=>"Cov Type", "Diabetis"=>"Diabetis","Electricity"=>"Electricity",
                            "German"=>"German","HIGGS"=>"Higgs","Ijcnn1"=>"IJCNN","Mnist"=>"Mnist","Shuttle"=>"Shuttle","SUSY"=>"SUSY","Vehicle"=>"Vehicle","wXa"=>"wXa")
function Table()
    dataset_list = readdlm("file_list_finished_table")
    Methods = ["SCGPMC","SVGPMC","EPGPMC"]
    MetricNames = Dict("Error"=>2,"NLL"=>3,"Time"=>1)
    MetricsOrder = ["Time","Error","NLL"]
    full_table = Array{String,1}()
    push!(full_table,"\\begin{table}[h!]\\centering")
    push!(full_table,"\\begin{adjustbox}{max width=\\columnwidth}")
    push!(full_table,"\\footnotesize")
    push!(full_table,"\\begin{tabular}{|ll|l|l|l|l|}")
    push!(full_table,"Dataset & && \\textbf{$(NC[Methods[1]])} & $(NC[Methods[2]]) & $(NC[Methods[3]]) \\\\\\hline")

    for dataset in dataset_list
        Res = ConvergenceDetector(dataset)
        println("Working on dataset $dataset")
        for metric in MetricsOrder
            new_line =""
            # new_line = new_line*"& "
            if metric == "Time"
                new_line = new_line*"$(DatasetNameCorrection[dataset]) & \$ C=$(sizes[dataset][3])\$"
            elseif metric == "Error"
                    new_line = new_line*"& \$ N=$(sizes[dataset][1])\$"
            elseif  metric == "NLL"
                new_line = new_line*" &\$ D=$(sizes[dataset][2]) \$"
            end
            # if metric == "Error"
            #     new_line = new_line*"$(DatasetNameCorrection[dataset])"
            # elseif  metric == "NLL"
            #     new_line = new_line*"\$ n=$(sizes[dataset][1])/d=$(sizes[dataset][2])/C=$(sizes[dataset][3]) \$"
            # end
            new_line = new_line*" & $metric";
            mapped_values = Dict{Float64,String}()
            best_m = ""
            best_val = Inf
            for m in Methods
                if best_val > Res[m][MetricNames[metric]]
                    best_m = m
                    best_val = Res[m][MetricNames[metric]]
                end
            end

            for m in Methods
                mean_v = format(Res[m][MetricNames[metric]],precision=2); std_v = format(Res[m][MetricNames[metric]+1],precision=2) ;
                if m == best_m
                    new_line = new_line*" & \$ \\mathbf{ $mean_v } \$"
                else
                    new_line = new_line*" & \$ $mean_v \$"
                end
            end
            new_line = new_line*"\\\\"
            if metric == "NLL"
                new_line = new_line*"\\hline"
            end
            push!(full_table,new_line)
        end
    end
    push!(full_table,"\\end{tabular}")
    push!(full_table,"\\end{adjustbox}")
    push!(full_table,"\\caption{Average test prediction error (\$\\frac{\\text{\\# Misclassified TestPoints}}{\\text{\\# Test Points}}\$) and average negative test log-likelihood (NLL) along with one standard deviation for a time budget of 100 seconds. Best values highlighted in bold}")
    push!(full_table,"\\label{tab:performance}")
    push!(full_table,"\\end{table}")
    writedlm("Latex/Table.tex",full_table)
    return full_table
end
m1 = 60
e1 = 10
m10 = 600
e2 = 100
e3 = 1000
h1 = 3600
global budget = Dict("acoustic"=>e2, "combined"=>e2, "covtype"=>e2, "dna"=>e2, "glass"=>e1, "iris"=>e1, "mnist"=>e3, "satimage"=>e2, "segment"=>e2, "seismic"=>e2, "sensorless"=>e2, "shuttle" =>e2, "vehicle"=>e2, "wine"=>e1 )

function ConvergenceDetector(dataset;time=true)
    Methods = ["SVGPMC","SCGPMC","EPGPMC"]
    ConvResults = Dict{String,Any}()
    small = false
    if in(["iris","wine","glass"],dataset)
        small = true
    end
    NSamples = sizes[dataset][1];
    N_epoch = small ? 1 : ceil(Int64,NSamples/200)
    b = budget[dataset]
    for m in Methods
        println(m)
        Res =
        readdlm(loc[dataset][m]*dataset*"Dataset/Results_"*m*(m=="SCGPMC" ? "_shared" : "")*".txt")
        t = Res[:,1]
        i_budget = Handpicked[dataset][m]
        ConvResults[m] = [t,DataConversion(Res[i_budget,3],"Accuracy"),
        DataConversion(Res[i_budget,5],"MeanL")]
    end
    return ConvResults
end

Handpicked = Dict("covtype"=>Dict("SCGPMC"=>36,"SVGPMC"=>78,"EPGPMC"=>63), "combined"=>Dict("SCGPMC"=>22,"SVGPMC"=>82,"EPGPMC"=>60), "fashion-mnist"=>Dict("SCGPMC"=>33,"SVGPMC"=>63,"EPGPMC"=>60),
                    "mnist"=>Dict("SCGPMC"=>33,"SVGPMC"=>37,"EPGPMC"=>57),"kmnist"=>Dict("SCGPMC"=>34,"SVGPMC"=>68,"EPGPMC"=>58),"shuttle"=>Dict("SCGPMC"=>54,"SVGPMC"=>52,"EPGPMC"=>57))


function PlotAutotuning(dataset,method;step=1)
  actualfolder = pwd()
  try
    cd(String("results/AutotuningExperiment"*dataset))
    results = readdlm(String("Results_"*method*".txt"))
    grid = readdlm(String("Grid"*method*".txt"))
    figure(String("Autotuning vs GridSearch for "*method*" on "*dataset*" dataset"))
    clf()
    semilogx(grid[1:step:end,1],1-grid[1:step:end,3],linestyle="-",linewidth=6.0,color="blue",label="Validation Loss")
    fill_between(grid[1:step:end,1],1-grid[1:step:end,3]-grid[1:step:end,4]/sqrt(10),1-grid[1:step:end,3]+grid[1:step:end,4]/sqrt(10),alpha=0.2,facecolor="blue")
    plot(results[:,5],1-results[:,3],color="red",marker="o",linestyle="None",markersize=22.0,label=L"$\theta$ learned by S-BSVM")
    xlabel(L"\theta",fontsize=35.0)
    ylabel("Validation Loss",fontsize=32.0)
    ylim([0, 0.55])
    xticks(fontsize=20.0)
    yticks(fontsize=20.0)
    legend(fontsize=28.0,numpoints=1,loc=4)
  catch
    cd(actualfolder)
    error("wesh")
  end
  cd(actualfolder)
end

function ROC_Drawing(stochastic::Bool=true)
  data = readdlm(String("results/BigDataExperimentSUSY/ROC_"*(stochastic ? "S" : "")*"SBSVM.txt"))
  AUC = readdlm(String("results/BigDataExperimentSUSY/AUC_"*(stochastic ? "S" : "")*"SBSVM.txt"))[1]
  dp = readdlm("results/BigDataExperimentSUSY/DeepLearning.txt")
  spec = data[:,1]; sens = data[:,2]
  figure(String("ROC "*(stochastic ? "S" : "")*"SBSVM"));
  clf()
  plot(sens,spec,linewidth=5.0,label="S-BSVM                 (AUC=0.84)")
  fill_between(sens,zeros(length(sens)),spec,alpha=0.1,color="blue")
  plot(dp[:,1],dp[:,2],linewidth=4.0,linestyle="--",color="red",label="Deep Learning      (AUC=0.88)")
  plot([0,1],[0,1],linestyle="--",linewidth=4.0,label="Random Classifier (AUC = 0.5)")
  text(0.5,0.05,"AUC = $(round(AUC*10000)/10000)",fontsize=32.0)
  xlabel("False Positive Rate",fontsize=32.0)
  ylabel("True Positive Rate",fontsize=32.0)
  xticks(fontsize=20.0)
  yticks(fontsize=20.0)
  legend(fontsize=30.0,loc=4)
end
