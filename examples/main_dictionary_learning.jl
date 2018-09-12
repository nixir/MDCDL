using MDCDL
using LinearAlgebra
using TestImages

using Base.Filesystem
using Dates

########## Configurations #########
# choose NSOLT type: (Rnsolt | Cnsolt)
Nsolt = Rnsolt
# TP := eltype(Nsolt) (<:AbstractFloat)
TP = Float64
# decimation factor: (<:NTuple{D,Int} where D is #dims)
df = (2,2)
# polyphase order: (<:NTuple{D,Int} where D)
ord = (4,4)
# number of channels: (<:Union{Integer,Tuple{Int,Int}} for Rnsolt)
#                     (<:Integer for Cnsolt)
nch = 8

# size of minibatches (<:NTuple{D,Int})
szx = (16,16)
# number of minibatches (<:Integer)
nSubData = 32

# path of log files (do nothing if isa(logdir, Nothing))
# tm = Dates.format(now(), "yyyy_mm_dd_SS_sss")
# logdir = joinpath(@__DIR__, "results", tm)
logdir = nothing

# options for sparse coding
sc_options = ( iterations = 1000, sparsity = 0.5, filter_domain=:convolution)
# options for dictionary update
du_options = ( iterations = 1, stepsize = 1e-3,)

# general options of dictionary learning
options = ( epochs  = 100,
            verbose = :standard,
            sc_options = sc_options,
            du_options = du_options,
            logdir = logdir,)
####################################

logdir != nothing && !isdir(logdir) && mkpath(logdir)

# original image
orgImg = TP.(testimage("cameraman"))

# generate minibatches of orgImage
trainingIds = map(1:nSubData) do nsd
    pos = rand.(UnitRange.(0,size(orgImg) .- szx))
    UnitRange.(1 .+ pos, szx .+ pos)
end
trainingSet = map(idx -> orgImg[idx...], trainingIds)

# create NSOLT instance
nsolt = Nsolt(TP, df, ord, nch)
# set initial matrices as random orthonormal ones.
MDCDL.rand!(nsolt, isPropMat = false, isPropAng = false, isSymmetry = false)

# dictionary learning
MDCDL.train!(nsolt, trainingSet; options...)
