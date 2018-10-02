using MLDataUtils, StatsBase, Random, Statistics
using DelimitedFiles

function convert_target!(data;end_pos=true)
    if end_pos
        y = data[:,end]
    else
        y = data[:,1]
    end
    y = convertlabel(LabelEnc.Indices(nlabel(y)),["$v" for v in y]).-1
    data = end_pos ? hcat(data[:,1:end-1],y) : hcat(data[:,2:end],y)
    return data
end

function normalizedata!(data;end_pos=true)
    if end_pos
        data[:,1] = std(data[:,1]) != 0 ?
        zscore(Float64.(data[:,1])) :
        zero(Float64.(data[:,1]))
    end
    for i in 2:(size(data,2)-1)
        data[:,i] = std(data[:,i]) != 0 ?
        zscore(Float64.(data[:,i])) :
        zero(Float64.(data[:,i]))
    end
    if !end_pos
        data[:,end] = std(data[:,end]) != 0 ?
        zscore(Float64.(data[:,end])) :
        zero(Float64.(data[:,end]))
    end
    return data
end


function shuffle_data!(data)
    nSamples = size(data,1)
    data = data[shuffle(1:nSamples),:]
    return data
end

function full_treatment(data,end_pos=true)
    data = normalizedata!(data,end_pos=end_pos)
    data = convert_target!(data,end_pos=end_pos)
    data = shuffle_data!(data)
    return data
end
