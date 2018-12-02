module MDCDL # Multi-Dimensional Convolutional Dictionary Learning

using LinearAlgebra
using ImageFiltering.Algorithm: Alg, FIR, FFT
using ComputationalResources: AbstractResource, CPU1

import Base: promote_rule, eltype, ndims, similar, length
import Random: rand, rand!

export rand, rand!

export Shapes, Optimizers, SparseCoders

export  PolyphaseVector, FilterBank, PolyphaseFB, AbstractNsolt,
        Cnsolt, CnsoltTypeI, CnsoltTypeII, Rnsolt, RnsoltTypeI, RnsoltTypeII,
        ParallelFilters, Multiscale, AbstractOperator, TransformSystems

export  istype1, istype2, permdctmtx, cdftmtx, analyze, synthesize,
        adjoint_synthesize, upsample, downsample, serialize, deserialize,
        analysisbank, decimations, orders, nchannels, nstages, kernels,
        analysiskernels, synthesiskernels, kernelsize, getrotations,
        setrotations!, mdarray2polyphase, polyphase2mdarray, iht,
        createTransform, loadfb, savefb, copy_params!

include("sparsecoders/SparseCoders.jl")
include("optimizers/Optimizers.jl")
using MDCDL.Optimizers: iterations

struct PolyphaseVector{T,D}
    data::AbstractMatrix{T}
    nBlocks::NTuple{D, Int}
end

promote_rule(::Type{PolyphaseVector{TA,D}}, ::Type{PolyphaseVector{TB,D}}) where {TA,TB,D} = PolyphaseVector{promote_type(TA,TB), D}

abstract type CodeBook{T,D} end
abstract type FilterBank{T,D} <: CodeBook{T,D} end
abstract type PolyphaseFB{T,D} <: FilterBank{T,D} end

eltype(::Type{CB}) where {T,D,CB<:CodeBook{T,D}} = T
ndims(::Type{CB}) where {T,D,CB<:CodeBook{T,D}} = D
ndims(cb::CodeBook) = ndims(typeof(cb))

decimations(fb::FilterBank) = fb.decimationFactor
nchannels(fb::FilterBank) = sum(fb.nChannels)
orders(fb::FilterBank) = fb.polyphaseOrder

kernels(fb::FilterBank) = fb.kernels
kernelsize(fb::FilterBank) = decimations(fb) .* (1 .+ orders(fb))

struct Multiscale
    filterbanks::Tuple
    Multiscale(fbs...) = new(fbs)
end

length(ms::Multiscale) = length(ms.filterbanks)

include("filterbanks/nsolt.jl")
include("filterbanks/parameterization.jl")
include("filterbanks/parallelFilters.jl")

include("orthonormalMatrixSystem.jl")
include("polyphaseMatrices.jl")

include("shapes.jl")
include("transformSystem.jl")

include("analysisSystem.jl")
include("synthesisSystem.jl")

include("dictionaryLearning/dictionaryLearning.jl")
include("dictionaryLearning/sparseCodingStage.jl")
include("dictionaryLearning/dictionaryUpdateStage.jl")
include("dictionaryLearning/logger.jl")

include("recipes.jl")
include("io.jl")

include("utils.jl")

end
