using UMAP, HDF5
using Plots
cd(dirname(@__FILE__))
data = h5read("../data/wine.h5","data")
X = copy(transpose(data[:,1:end-1]))
y = Int64.(data[:,end])
X_embed = umap(X,2)
p = plot()
for i in 0:length(unique(y))-1
    scatter!(X_embed[1,y.==i],X_embed[2,y.==i],lab="$i",markerstrokewidth=0)
end
display(p);
