using MDCDL
using LinearAlgebra
using Images, TestImages
using Plots

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
ord = (2,2)
# number of channels: (<:Union{Integer,Tuple{Int,Int}} for Rnsolt)
#                     (<:Integer for Cnsolt)
nch = 6
# number of tree level (<: Integer)
level = 2

# size of minibatches (<:NTuple{D,Int})
szx = (16,16)
# number of minibatches (<:Integer)
nSubData = 16

# path of log files (do nothing if isa(logdir, Nothing))
# logdir = joinpath(@__DIR__, "results", Dates.format(now(), "yyyymmdd_HH_MM_SS_sss"))
logdir = nothing
# save minibatches?
do_save_trainingset = false
do_export_atoms = false

# options for sparse coding
# sparsecoder = SparseCoders.IHT( iterations = 1000, sparsity = 0.5, filter_domain=:convolution)
sparsecoder = SparseCoders.IHT(iterations = 100, nonzeros=trunc(Int,0.4*prod(szx)))
# options for dictionary update
optimizer = Optimizers.Steepest(iterations = 100, rate = 1e-3)
# optimizer = Optimizers.Adam( iterations = 1000)

# general options of dictionary learning
options = ( epochs  = 1000,
            verbose = :standard, # :none, :standard, :specified, :loquacious
            sparsecoder = sparsecoder,
            optimizer = optimizer,
            logdir = logdir,)
####################################
logdir != nothing && !isdir(logdir) && mkpath(logdir)

# original image
orgImg = testimage("cameraman")

# generate minibatches of orgImage
trainingIds = map(1:nSubData) do nsd
    pos = rand.(UnitRange.(0,size(orgImg) .- szx))
    UnitRange.(1 .+ pos, szx .+ pos)
end
trainingSet = map(idx -> orgImg[idx...], trainingIds)
if logdir != nothing && do_save_trainingset
    datadir = joinpath(logdir, "data")
    !isdir(datadir) && mkpath(datadir)
    map(idx->save(joinpath(datadir, "$idx.png"), trainingSet[idx]), 1:nSubData)
end

# create NSOLT instance
nsolt = Nsolt(TP, df, ord, nch)
# set random orthonormal matrices to the initial matrices.
# MDCDL.rand!(nsolt, isPropMat = false, isPropAng = false, isSymmetry = false)
MDCDL.rand!(nsolt)
nsolt.initMatrices[1] .= cat(1, qr(rand((size(nsolt.initMatrices[1]) .- 1)...)).Q, dims=[1,2])

msnsolt = Multiscale([ similar(nsolt) for l in 1:level ]...)

# dictionary learning
MDCDL.train!(msnsolt, trainingSet; options...)

if logdir != nothing && do_export_atoms
    for idx = 1:length(msnsolt)
        png(plot(msnsolt[idx]), joinpath(logdir, string("nsolt_",idx,".png")))
    end
end
