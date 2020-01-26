
#Load some packages and add workers
using Images
using Distributed
using NPZ
using LinearAlgebra
using Statistics
include("../../src/DPMMSubClusters.jl")
#Load the package on all workers
img = npzread("./files/f2.npy")
input_arr = transpose(img)
conn_components = npzread("./files/ConnectedComponents2.npy")

#Create HyperParams
#The rgb,xy multiplier allows us to both play with the weight of the xy/rgb
rgb_prior_multiplier = 1
xy_prior_multiplier = 0.4
flow_prior_multiplier = 1

data_cov = cov(input_arr')
data_cov[4:5,1:3] .= 0
data_cov[1:3,4:5] .= 0
data_cov[1:3,6:7] .= 0
data_cov[6:7,1:3] .= 0
data_cov[6:7,4:5] .= 0
data_cov[4:5,6:7] .= 0


data_cov[1:3,1:3] .*= rgb_prior_multiplier
data_cov[4:5,4:5] .*= xy_prior_multiplier
data_cov[6:7,6:7] .*= flow_prior_multiplier

data_mean = mean(input_arr,dims = 2)[:]

hyper_params = DPMMSubClusters.niw_hyperparams(1.0,
           data_mean,
           8,
           data_cov)

#Run the model
labels,clusters,weights = DPMMSubClusters.fit(input_arr,hyper_params,50000.0,iters = 150, verbose = true, conn_components = conn_components)
# npzwrite("./files/labels2.npy", labels)

#
# #Get the cluster color means
# color_means = [x.μ[1:3] for x in clusters]
#
#
# segemnated_image = zeros(3,x,y)
# for i=1:x
#     for j=1:y
#         segemnated_image[:,i,j] = color_means[labels[(i-1)*y+j]]
#     end
# end
# segemnated_image = colorview(RGB,segemnated_image)
