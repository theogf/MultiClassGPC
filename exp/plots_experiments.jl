using PyPlot
using Formatting
using PyCall
plt[:style][:use]("seaborn-colorblind")

if VERSION >= v"0.7.0-"
    using DelimitedFiles
end
NC =  Dict("EPGPMC"=>"EPGPMC","TTGPC"=>"Tensor Train GPC", "LogReg"=>"Linear Model",
"SVGPMC"=>"SVGPMC","SXGPMC"=>"X-GPC","Accuracy"=>"Avg. Test Error",
"MedianL"=>"Avg. Median Neg.\n Test Log-Likelihood","MeanL"=>"Avg. Mean Neg.\n Test Log-Likelihood")
colors=Dict("SVGPMC"=>"blue","SXGPMC"=>"red","LogReg"=>"yellow","EPGPMC"=>"green", "TTGPC"=>"black")
linestyles=Dict(16=>":",32=>"--",64=>"-.",128=>"-")
markers=Dict(16=>"",32=>"o",64=>"x",128=>"+")
# linestyles=Dict(4=>"-",8=>":",10=>"-",16=>"-.",32=>"--",50=>":",64=>"-.",100=>"-.",128=>"-",200=>"--",256=>"--")
metrics = Dict("Accuracy"=>3,"MeanL"=>5,"MedianL"=>7,"ELBO"=>9)

# location of the results
c = "../cluster/AT_Experiment/"
cs = "../cluster/AT_S_Experiment/"
l = "results/Experiment/"
las = "results/AT_S_Experiment/"
ls = "results/S_Experiment/"
f = "../final_results/"
loc = Dict{String,Dict{String,String}}()
loc["iris"] =           Dict("SXGPMC"=>l,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["wine"] =           Dict("SXGPMC"=>l,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["glass"] =          Dict("SXGPMC"=>l,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["vehicle"] =          Dict("SXGPMC"=>cs,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["dna"] =          Dict("SXGPMC"=>cs,"SVGPMC"=>cs,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["satimage"] =          Dict("SXGPMC"=>ls,"SVGPMC"=>ls,"EPGPMC"=>cs,"TTGPC"=>cs)
loc["segment"] =          Dict("SXGPMC"=>ls,"SVGPMC"=>ls,"EPGPMC"=>ls,"TTGPC"=>cs)
loc["Cod-rna"] =            Dict("SXGPMC"=>c,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Covtype"] =            Dict("SXGPMC"=>c,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Credit_card"] =        Dict("SXGPMC"=>c,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Diabetis"] =           Dict("SXGPMC"=>l,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["Electricity"] =        Dict("SXGPMC"=>l,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["German"] =             Dict("SXGPMC"=>l,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["HIGGS"] =              Dict("SXGPMC"=>c,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Ijcnn1"] =             Dict("SXGPMC"=>c,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Mnist"] =              Dict("SXGPMC"=>l,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["Poker"] =              Dict("SXGPMC"=>c,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Protein"] =            Dict("SXGPMC"=>c,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Shuttle"] =            Dict("SXGPMC"=>l,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)
loc["SUSY"] =               Dict("SXGPMC"=>c,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["Vehicle"] =            Dict("SXGPMC"=>c,"SVGPMC"=>c,"EPGPMC"=>c,"TTGPC"=>c)
loc["wXa"] =                Dict("SXGPMC"=>l,"SVGPMC"=>l,"EPGPMC"=>l,"TTGPC"=>c)


gwidth = 2.0
gmarkersize= 5.0
function DataConversion(array,name)
    if name == "Accuracy"
        return 1.0.-array
    elseif name == "MedianL"
        return -array
    elseif name == "MeanL"
        return -array
    end
end

function InducingPointsComparison(metric,MPoints=[16,32,64,128];step=1)
    dataset="Shuttle"
    figure("Comparison of inducing points accuracy and time",figsize=(16,8)); clf();
    p = Dict("SVGPMC"=>Array{Any,1}(),"SXGPMC"=>Array{Any,1}(),"EPGPMC"=>Array{Any,1}())
    lab = Dict("SVGPMC"=>Array{Any,1}(),"SXGPMC"=>Array{Any,1}(),"EPGPMC"=>Array{Any,1}())
    for M in MPoints
        Results = Dict{String,Any}()
        Results["SXGPMC"] = readdlm("data_M$M/ConvergenceExperiment_AT/$(dataset)Dataset/Results_SXGPMC.txt")
        Results["SVGPMC"] = readdlm("data_M$M/ConvergenceExperiment_AT/$(dataset)Dataset/Results_SVGPMC.txt")
        Results["EPGPMC"] = readdlm("data_M$M/ConvergenceExperiment_AT/$(dataset)Dataset/Results_EPGPMC.txt")
        Results["EPGPMC"][:,1] += Float64(readdlm("results/time_correction"*"/$(dataset)_M$(M)")[1])
        for (name,res) in Results
            res[:,metrics[metric]] = DataConversion(res[:,metrics[metric]],metric)
            push!(lab[name],NC[name]*" M=$M")
            new_p,=semilogx(res[1:step:end,1],SmoothIt(res[1:step:end,metrics[metric]],window=3),markersize=gmarkersize,color=colors[name],marker=markers[M],linewidth=gwidth,linestyle=linestyles[M],label=NC[name]*" M=$M")
            push!(p[name],new_p)
        end
    end


    xlabel("Training Time in Seconds",fontsize=20.0)
    xticks(fontsize=18.0)
    ylabel(NC[metric],fontsize=20.0)
    yticks(fontsize=18.0)
    title(dataset,fontsize=24.0,fontweight="semibold")

    legend([p["SXGPMC"];p["SVGPMC"];p["EPGPMC"]],[lab["SXGPMC"];lab["SVGPMC"];lab["EPGPMC"]],fontsize=18.0)
    xlim([0.03,4500])
    ylim([-0.01,0.15])
    tight_layout()
    savefig("../../Plots/$(dataset)InducingPointsPlot.png")
end

function PlotAll()
    file_list = readdlm("file_list_finished")
    for file in file_list
        PlotMetricvsTime(file,"Final",time=true,writing=true,corrections=true)
    end
end

function SmoothIt(x;window=3)
    smoothed = zero(x)
    for i in 1:length(x)
        smoothed[i] = mean(x[max(1,i-window):min(length(x),i+window)])
    end
    return smoothed
end

function PlotMetricvsTime(dataset,metric;final=false,AT=true,time=true,writing=false,corrections=false)
    Results = Dict{String,Any}();
    println("Working on dataset $dataset")
    # colors=Dict("GPC"=>"b","SPGGPC"=>"r","LogReg"=>"y")
    time_line = [1:1:99;100:10:999;1000:100:9999;10000:1000:40000]
    # Dict("SVGPMC"=>[1:1:99;100:10:999;1000:100:9999;10000:1000:20000],"SXGPMC"=>[1:1:99;100:10:999;1000:100:20000])
    p = Dict{String,Any}()

    FinalMetrics = ["MeanL","Accuracy"]


    # NC =  Dict("LogReg"=>"Linear Model","GPC"=>"SVGPMC","SPGGPC"=>"X-GPC","Accuracy"=>"Avg. Test Error","MedianL"=>"Avg. Median Neg. Test Log likelihood")
    Results["SVGPMC"] = readdlm(loc[dataset]["SVGPMC"]*dataset*"Dataset/Results_SVGPMC.txt")
    Results["SXGPMC"] = readdlm(loc[dataset]["SXGPMC"]*dataset*"Dataset/Results_SXGPMC.txt")
    Results["EPGPMC"] = readdlm(loc[dataset]["EPGPMC"]*dataset*"Dataset/Results_EPGPMC.txt")
    # Results["EPGPMC"][:,1] = Results["EPGPMC"][:,1] + Float64(readdlm(loc[dataset]["EPGPMC"]*"/../time_correction/"*dataset)[1])
    # Results["TTGPC"] = readdlm(loc[dataset]["TTGPC"]*dataset*"Dataset/Results_TTGPC.txt")
    # Results["LogReg"] = readdlm(loc[dataset]["LogReg"]*dataset*"Dataset/Results_LogReg.txt")

    maxx = max(Results["SVGPMC"][end,1],Results["EPGPMC"][end,1],Results["SXGPMC"][end,1])#,Results["TTGPC"][end,1])
    minx = min(Results["SVGPMC"][1,1],Results["EPGPMC"][1,1],Results["SXGPMC"][1,1])#,Results["TTGPC"][1,1])
    # logreg = zeros(2,10)
    # logreg[1,:] = Results["LogReg"]; logreg[2,2:end] = Results["LogReg"][2:end]; logreg[2,1] = maxx;
    # println(logreg)
    # Results["LogReg"] = logreg
    if metric != "Final"
        figure("Convergence on dataset "*dataset*" ",figsize=(16,9));clf();
    else
        figure("Convergence on dataset "*dataset*" ",figsize=(13.23,4.7));clf();
    end
    step=1
    if corrections
        if dataset == "aXa"
            #Divide acc stderr by 2
            Results["SXGPMC"][:,4] *= 0.5;
            Results["SVGPMC"][:,4] *= 0.5;
            Results["EPGPMC"][:,4] *= 0.5;
        elseif dataset == "Bank_marketing"
            #Divide acc stderr by 2
            Results["SXGPMC"][:,4] *= 0.5;
            Results["SVGPMC"][:,4] *= 0.5
            Results["EPGPMC"][:,4] *= 0.5
            Results["SXGPMC"][:,6] *= 0.5;
            Results["SVGPMC"][:,6] *= 0.5
            Results["EPGPMC"][:,6] *= 0.5
        elseif dataset == "Electricity"
            #Divide acc stderr by 2
            Results["SXGPMC"][:,4] *= 0.5;
            Results["SVGPMC"][:,4] *= 0.5
            Results["EPGPMC"][:,4] *= 0.5
        elseif dataset == "German"
            Results["SXGPMC"][:,4] *= 0.5;
            Results["SVGPMC"][:,4] *= 0.5;
            Results["EPGPMC"][:,4] *= 0.5;
            Results["SXGPMC"][:,6] *= 0.5;
            Results["SVGPMC"][:,6] *= 0.5;
            Results["EPGPMC"][:,6] *= 0.5;
        end
    end
    if time
        time_line = Dict("SVGPMC"=>Results["SVGPMC"][:,1],"SXGPMC"=>Results["SXGPMC"][:,1],"EPGPMC"=>Results["EPGPMC"][:,1])#,"LogReg"=>Results["LogReg"][:,1],"TTGPC"=>Results["TTGPC"][:,1])
    else
        time_line = Dict("SVGPMC"=>time_line[1:length(Results["SVGPMC"][:,1])],"SXGPMC"=>time_line[1:length(Results["SXGPMC"][:,1])],
        "EPGPMC"=>time_line[1:length(Results["EPGPMC"][:,1])])#,"LogReg"=>[1,10000],"TTGPC"=>time_line[1:length(Results["TTGPC"][:,1])])
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
                legend([p["SXGPMC"];p["SVGPMC"];p["EPGPMC"]],#;p["TTGPC"];p["LogReg"]],
                [NC["SXGPMC"];NC["SVGPMC"];NC["EPGPMC"]],fontsize=20.0)#;NC["TTGPC"];NC["LogReg"]])
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
            subplot(2,2,giter)
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
        savefig("../../Plots/"*(metric=="Final" ? "Final" : "")*"Convergence_vs_"*(time ? "time" : "iterations")*"_on_"*dataset*".png")
        close()
    end
end


sizes = Dict("aXa"=>(36974,123),"Bank_marketing"=>(45211,43),"Click_Prediction"=>(399482,12),"Cod-rna"=>(343564,8),"Covtype"=>(581012,54),
                    "Diabetis"=>(768,8),"Electricity"=>(45312,8),"German"=>(1000,20),"HIGGS"=>(11000000,28),"Ijcnn1"=>(141691,22),"Mnist"=>(70000,780),"Poker"=>(1025010,10),
                    "Protein"=>(24837,357),"Shuttle"=>(58000,9),"SUSY"=>(5000000,18),"Vehicle"=>(98528,100),"wXa"=>(34780,300))
DatasetNameCorrection = Dict("iris"=>"Iris","wine"=>"Wine","glass"=>"Glass","vehicle"=>"Vehicle", "segment"=>"Segment", "dna"=>"DNA","satimage"=>"SatImage",
                            "Cod-rna"=>"Cod RNA", "Covtype"=>"Cov Type", "Diabetis"=>"Diabetis","Electricity"=>"Electricity",
                            "German"=>"German","HIGGS"=>"Higgs","Ijcnn1"=>"IJCNN","Mnist"=>"Mnist","Shuttle"=>"Shuttle","SUSY"=>"SUSY","Vehicle"=>"Vehicle","wXa"=>"wXa")
function Table()
    dataset_list = readdlm("file_list_finished")
    Methods = ["SXGPMC","SVGPMC","EPGPMC"]
    MetricNames = Dict("Error"=>3,"NLL"=>5,"Time"=>1)
    MetricsOrder = ["Error","NLL","Time"]
    full_table = Array{String,1}()
    first_line = String("\\begin{table}[h!]\\centering
\\begin{adjustbox}{max width=\\columnwidth}
\\footnotesize
\\begin{tabular}{|l|l|l|l|l|}
Dataset & & \\textbf{$(NC[Methods[1]])} & $(NC[Methods[2]]) & $(NC[Methods[3]]) \\\\\\hline")
    push!(full_table,first_line)

    for dataset in dataset_list
        Res = ConvergenceDetector(dataset,"Accuracy",epsilon=1e-3,window=10)
        println("Working on dataset $dataset")
        for metric in MetricsOrder
            new_line =""
            # new_line = new_line*"& "
            if metric == "Error"
                new_line = new_line*"\\multirow{1}{*}{$(DatasetNameCorrection[dataset])}"
            elseif  metric == "NLL"
                new_line = new_line*"\$ n=$(sizes[dataset][1]) \$"
            else
                new_line = new_line*"\$ d=$(sizes[dataset][2]) \$"
            end
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
                if metric != "Time"
                    mean_v = format(Res[m][MetricNames[metric]],precision=2); std_v = format(Res[m][MetricNames[metric]+1],precision=2) ;
                else
                    # if Res[m][MetricNames[metric]] > 1000
                    #     mean_v = fmt(FormatSpec(".2e"),floor(Int64,Res[m][MetricNames[metric]]));
                    if Res[m][MetricNames[metric]] > 10
                        mean_v = format(floor(Int64,Res[m][MetricNames[metric]]));
                    elseif Res[m][MetricNames[metric]] > 1
                        mean_v = format(Res[m][MetricNames[metric]],precision=1);
                    else
                        mean_v = format(Res[m][MetricNames[metric]],precision=2);
                    end
                    if Res[m][MetricNames[metric]+1] > 10
                        std_v = format(floor(Int64,Res[m][MetricNames[metric]+1]));
                    elseif Res[m][MetricNames[metric]+1] > 1
                        std_v = format(Res[m][MetricNames[metric]+1],precision=1);
                    else
                        std_v = format(Res[m][MetricNames[metric]+1],precision=2);
                    end
                end
                if m == best_m
                    new_line = new_line*" & \$ \\mathbf{ $mean_v \\pm $std_v } \$"
                else
                    new_line = new_line*" & \$ $mean_v \\pm $std_v \$"
                end
            end
            new_line = new_line*"\\\\"
            if metric == "Time"
                new_line = new_line*"\\hline"
            end
            push!(full_table,new_line)
        end
    end
    last_line = "\\end{tabular}
\\end{adjustbox}
\\caption{Average test prediction error, negative test log-likelihood (NLL) and time in seconds along with one standard deviation.
}
\\label{tab:performance}
\\end{table}"
    push!(full_table,last_line)
    writedlm("Latex/Table.tex",full_table)
    return full_table
end
function ConvergenceDetector(dataset,metric;epsilon=1e-3,window=5,plot=false)
    Methods = ["SVGPMC","SXGPMC","EPGPMC"]
    ConvResults = Dict{String,Any}()
    ConvIter = Dict{String,Any}()
    figure(2);clf()
    giter=1
    for m in Methods
        Res =
        readdlm(loc[dataset][m]*dataset*"Dataset/Results_"*m*".txt")
        if m == "EPGPMC"
            corr = Float64(readdlm(loc[dataset]["EPGPMC"]*"/../time_correction/"*dataset)[1])
        end
        t = Res[:,1]
        values = DataConversion(Res[:,metrics[metric]],metric)
        Res[:,3] = SmoothIt(DataConversion(Res[:,3],"Accuracy"),window)
        Res[:,5] = SmoothIt(DataConversion(Res[:,5],"MedianL"),window)
        converged = false
        values = SmoothIt(values,window)
        conv = abs.(values[2:end]-values[1:end-1])
        if haskey(Handpicked,dataset) && haskey(Handpicked[dataset],m)
            ConvResults[m] = Res[Handpicked[dataset][m],1:6]
        else
            iter=2
            if m == "SVGPMC"
                iter = 200
            end
            while iter <= length(conv)
                if mean(conv[max(1,iter-window):min(length(conv),iter+window)])<epsilon
                # if conv[iter] < epsilon
                    ConvResults[m] = Res[iter,1:6];
                    ConvIter[m] = iter
                    converged = true
                    break;
                end
                iter+=1
            end
            if !converged
                ConvResults[m] = Res[end,1:6]
                ConvIter[m] = size(Res,1)
                println("Reached end for dataset $dataset with method $m")
            end
            if m == "EPGPMC"
                ConvResults[m][1] += corr
            end
        end
    end
    if plot
        display(ConvIter)
        PlotMetricvsTime(dataset,"Accuracy")
        for m in Methods
            scatter([ConvResults[m][1]],[1-ConvResults[m][3]],color=colors[m],s=300.0)
        end
    end
    return ConvResults
end

Handpicked = Dict("aXa"=>Dict("EPGPMC"=>197), "Bank_marketing"=>Dict(), "Click_Prediction"=>Dict(),
                    "Cod-rna"=>Dict(),"Covtype"=>Dict(),"Diabetis"=>Dict("SXGPMC"=>82,"SVGPMC"=>236),"Electricity"=>Dict(),
                    "German"=>Dict("SXGPMC"=>86,"SVGPMC"=>282),"HIGGS"=>Dict(),"Ijcnn1"=>Dict(),"Mnist"=>Dict(),"Shuttle"=>Dict(),
                    "SUSY"=>Dict(),"wXa"=>Dict())

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
