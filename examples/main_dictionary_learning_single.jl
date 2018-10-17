using MDCDL
using LinearAlgebra
using Images, TestImages
using Plots

using Base.Filesystem
using Dates
using NLopt

########## Configurations #########
# choose NSOLT type: (Rnsolt | Cnsolt)
Nsolt = Cnsolt
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
szx = (64,64)

# path of log files (do nothing if isa(logdir, Nothing))
# logdir = joinpath(@__DIR__, "results", Dates.format(now(), "yyyymmdd_HH_MM_SS_sss"))
logdir = nothing
# save minibatches?
do_save_trainingset = false
do_export_atoms = false

# options for sparse coding
sparsity = 0.2
sparsecoder = SparseCoders.IHT(iterations = 100, nonzeros = trunc(Int, sparsity * prod(szx)), filter_domain=:convolution)
# options for dictionary update
# optimizer = Optimizers.Steepest(iterations = 100, rate = 1e-4 )
optimizer = Optimizers.GlobalOpt(iterations=500)
# optimizer = Optimizers.AdaGrad(iterations = 30)

# general options of dictionary learning
options = ( epochs  = 20,
            verbose = :standard, # :none, :standard, :specified, :loquacious
            sparsecoder = sparsecoder,
            optimizer = optimizer,
            logdir = logdir,)
####################################
logdir != nothing && !isdir(logdir) && mkpath(logdir)

# original image
orgImg = testimage("cameraman")

# generate minibatches of orgImage
trainingSet = [ imresize(orgImg, szx) .|> Float64 ]

# create NSOLT instance
nsolt = Nsolt(TP, df, ord, nch)
# set random orthonormal matrices to the initial matrices.
# MDCDL.rand!(nsolt, isPropMat = false, isPropAng = false, isSymmetry = false)

# dictionary learning
MDCDL.train!(nsolt, trainingSet; options...)

if logdir != nothing && do_export_atoms
    png(plot(nsolt), joinpath(logdir, "atoms.png"))
end
