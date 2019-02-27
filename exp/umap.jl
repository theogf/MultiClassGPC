using UMAP, HDF5
using Plots
pyplot()
plotlyjs()
cd(dirname(@__FILE__))
dataset = "segment"
file_loc = "../data/"*dataset*".h5"

##
    # data = h5read("../data/wine.h5","data")
    # X = copy(transpose(data[:,1:end-1]))
    # y = Int64.(data[:,end])
    X = vcat(h5read(file_loc,"data/X_train"),h5read(file_loc,"data/X_test"))
    y = vcat(h5read(file_loc,"data/y_train"),h5read(file_loc,"data/y_test"))
X_embed = umap(X',3)
p = plot()
for i in 0:length(unique(y))-1
    # scatter!(X_embed[1,y.==i],X_embed[2,y.==i],lab="$i",markerstrokewidth=0)
    scatter3d!(p,X_embed[1,y.==i],X_embed[2,y.==i],X_embed[3,y.==i],lab="$i",markerstrokewidth=0)
end
gui()
p= plot()
for i in 0:length(unique(y))-1
    # scatter!(X_embed[1,y.==i],X_embed[2,y.==i],lab="$i",markerstrokewidth=0)
    scatter3d!(X'[1,y.==i],X'[2,y.==i],X'[12,y.==i],lab="$i",markerstrokewidth=0)
end
gui()
