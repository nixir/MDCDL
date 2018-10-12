# MDCDL [![Build Status](https://travis-ci.org/sngy/MDCDL.svg?branch=master)](https://travis-ci.org/sngy/MDCDL)

Install
-------

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/sngy/MDCDL.git",rev="master"))
```

Example
-------

dictionary learning of Nonseparable Oversampled Lapped Transforms (NSOLTs)

```julia
using MDCDL
using LinearAlgebra
using Images, TestImages

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
nch = 8
# number of tree level (<: Integer)
level = 3

# size of minibatches (<:NTuple{D,Int})
szx = (16,16)
# number of minibatches (<:Integer)
nSubData = 16

# options for sparse coding
sparsecoder = SparseCoders.IHT(iterations = 100, nonzeros=trunc(Int,0.2*prod(szx)))
# options for dictionary update
optimizer = Optimizers.Steepest(iterations = 1, rate = 1e-4)

# general options of dictionary learning
options = ( epochs  = 1000,
            verbose = :standard, # :none, :standard, :specified, :loquacious
            sparsecoder = sparsecoder,
            optimizer = optimizer)
####################################
# original image
orgImg = testimage("cameraman")

# generate a set of minibatch of original image.
trainingIds = map(1:nSubData) do nsd
    pos = rand.(UnitRange.(0,size(orgImg) .- szx))
    UnitRange.(1 .+ pos, szx .+ pos)
end
trainingSet = map(idx -> orgImg[idx...], trainingIds)

# create NSOLT instance
nsolt = Nsolt(TP, df, ord, nch)
# set random orthonormal matrices to the initial matrices.
MDCDL.rand!(nsolt, isPropMat = false, isPropAng = false, isSymmetry = false)
istype1(nsolt) && MDCDL.vm1constraint!(nsolt)

msnsolt = Multiscale([ deepcopy(nsolt) for l in 1:level ]...)

# dictionary learning
MDCDL.train!(msnsolt, trainingSet; options...)

# If you want to show the atomic images, uncomment below line (requires Plots.jl)
# plot(map(ns->plot(ns), msnsolt.filterbanks)...)

```
