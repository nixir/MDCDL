module MDCDL # Multi-Dimensional Convolutional Dictionary Learning

using LinearAlgebra
using ImageFiltering
using ImageFiltering.Algorithm: Alg, FIR, FFT
using ComputationalResources: AbstractResource, CPU1

import Base: promote_rule, eltype, ndims, similar, length
import Random: rand, rand!

export rand, rand!

export PolyphaseVector
export FilterBank, PolyphaseFB, AbstractNsolt, Cnsolt, CnsoltTypeI, CnsoltTypeII, Rnsolt, RnsoltTypeI, RnsoltTypeII, ParallelFilters
export Multiscale

export istype1, istype2, permdctmtx, cdftmtx
export analyze, synthesize, adjoint_synthesize
export upsample, downsample
export serialize, deserialize
export analysisbank
export decimations, orders, nchannels
export kernels, analysiskernels, synthesiskernels, kernelsize
export getrotations, setrotations!
export mdarray2polyphase, polyphase2mdarray
export iht
export AbstractOperator
export TransformSystem
export createTransform
export Shapes, Optimizers, SparseCoders
export loadfb, savefb

include("sparsecoders/SparseCoders.jl")
include("optimizers/Optimizers.jl")
using MDCDL.Optimizers: iterations

module Shapes
    abstract type AbstractShape end

    struct Separated <: AbstractShape
        Separated(sz...) = new()
    end
    struct Vectorized <: AbstractShape
        insize::Tuple
        Vectorized(sz::Integer...) = new(sz)
        Vectorized(sz::AbstractVector) = Vectorized(sz...)
        Vectorized(sz::Tuple) = Vectorized(sz...)
    end
    struct Combined <: AbstractShape
        Combined(sz...) = new()
    end

    Vec = Vectorized
end

isfixedsize(::A) where {A<:Shapes.AbstractShape} = isfixedsize(A)
isfixedsize(::Type{S}) where {S<:Shapes.AbstractShape}= false
isfixedsize(::Type{Shapes.Vec}) = true

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

struct ParallelFilters{T,D} <: FilterBank{T,D}
    decimationFactor::NTuple{D,Int}
    polyphaseOrder::NTuple{D,Int}
    nChannels::Integer

    # kernels::Vector{AbstractArray{T,D}}
    kernelspair::NTuple{2}

    function ParallelFilters(ker::NTuple{2,Vector{A}}, df::NTuple{D,Int}, ord::NTuple{D,Int}, nch::Integer) where {T,D,A<:AbstractArray{T,D}}
        new{T,D}(df, ord, nch, ker)
    end

    function ParallelFilters(ker::Vector{A}, df::Tuple, ord::Tuple, nch::Integer) where {T,D,A<:AbstractArray{T,D}}
        ParallelFilters((ker,ker), df, ord, nch)
    end

    function ParallelFilters(::Type{T}, df::NTuple{D,Int}, ord::NTuple{D,Int}, nch::Integer) where {T,D}
        new{T,D}(df, ord, nch, ([fill(zeros(df .* (ord .+ 1)), nch) for idx=1:2 ]...,))
    end

    function ParallelFilters(fb::FilterBank)
        ParallelFilters(kernels(fb), decimations(fb), orders(fb), nchannels(fb))
    end
end

analysiskernels(pf::ParallelFilters) = kernelspair[1]
synthesiskernels(pf::ParallelFilters) = kernelspair[2]
kernels(pf::ParallelFilters) = kernelspair

struct Multiscale
    filterbanks::Tuple
    Multiscale(fbs...) = new(fbs)
end

length(ms::Multiscale) = length(ms.filterbanks)

abstract type AbstractOperator end

struct TransformSystem{OP} <: AbstractOperator
    shape::Shapes.AbstractShape
    operator::OP
    options::Base.Iterators.Pairs

    function TransformSystem(operator::OP, shape=Shapes.Separated(); options...) where {OP<:FilterBank}
        new{OP}(shape, operator, options)
    end

    function TransformSystem(ts::TransformSystem, shape=ts.shape)
        TransformSystem(deepcopy(ts.operator), shape; ts.options...)
    end
end

decimations(tfs::TransformSystem) = decimations(tfs.operator)
orders(tfs::TransformSystem) = orders(tfs.operator)
nchannels(tfs::TransformSystem) = nchannels(tfs.operator)

createTransform(ns::FilterBank, args...; kwargs...) = TransformSystem(ns, args...; kwargs...)

struct JoinedTransformSystems{T} <: AbstractOperator
    shape::Shapes.AbstractShape
    transforms::Array

    JoinedTransformSystems(ts::Tuple{TS}, args...; kwargs...) where{TS<:TransformSystem} = JoinedTransformSystems(Multiscale(ts...), args...; kwargs...)
    function JoinedTransformSystems(mst::MS, shape=Shapes.Separated()) where {TS<:TransformSystem,MS<:Multiscale}
        new{MS}(shape, collect(mst.filterbanks))
    end
end

function createTransform(ms::MS, shape::S=Shapes.Separated()) where {MS<:Multiscale,S<:Shapes.AbstractShape}
    opsarr = map(1:length(ms.filterbanks)) do lv
        sp = if isfixedsize(S)
            S(fld.(shape.insize, decimations(ms.filterbanks[lv]).^(lv-1)))
        else
            S()
        end
        TransformSystem(ms.filterbanks[lv], sp)
    end
    JoinedTransformSystems(MS(opsarr...), shape)
end

include("nsolt.jl")

include("orthonormalMatrixSystem.jl")
include("polyphaseMatrices.jl")


include("analysisSystem.jl")
include("synthesisSystem.jl")
include("transforms.jl")

include("parameterizeFilterBanks.jl")

include("dictionaryLearning.jl")

include("recipes.jl")
include("io.jl")

include("utils.jl")

end
