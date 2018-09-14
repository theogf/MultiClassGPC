using MLDataUtils, StatsBase, Random

function convert_target!(data;end_pos=true)
    if end_pos
        y = data[:,end]
    else
        y = data[:,1]
    end
    y = convertlabel(LabelEnc.Indices(nlabel(y)),y).-1
    data = end_pos ? hcat(data[:,1:end-1],y) : hcat(data[:,2:end],y)
    println(data)
    return data
end

function normalize!(data;end_pos=true)
    if end_pos
        data[:,1] = zscore(Float64.(data[:,1]))
    end
    for i in 2:(size(data,2)-1)
        data[:,i] = zscore(Float64.(data[:,i]))
    end
    if !end_pos
        data[:,end] = zscore(Float64.(data[:,end]))
    end
    return data
end

function shuffle_data!(data)
    nSamples = size(data,1)
    data = data[shuffle(1:nSamples),:]
    return data
end

function full_treatment(data,end_pos=true)
    data = normalize!(data)
    data = convert_target!(data,end_pos=end_pos)
    data = shuffle_data!(data)
    return data
end
