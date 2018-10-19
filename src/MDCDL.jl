module MDCDL # Multi-Dimensional Convolutional Dictionary Learning

using LinearAlgebra
using ImageFiltering
using ImageFiltering.Algorithm: Alg, FIR, FFT
using ComputationalResources: AbstractResource, CPU1

import Base: promote_rule, eltype, ndims, similar, length
import Random: rand, rand!

export rand, rand!

export PolyphaseVector
export FilterBank, PolyphaseFB, AbstractNsolt, Cnsolt, Rnsolt, ParallelFilters
export Multiscale, MultiLayerCsc

export istype1, istype2
export analyze, synthesize, adjoint_synthesize
export upsample, downsample
export permdctmtx, cdftmtx
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
abstract type AbstractNsolt{T,D} <: PolyphaseFB{T,D} end

eltype(::Type{CB}) where {T,D,CB<:CodeBook{T,D}} = T
ndims(::Type{CB}) where {T,D,CB<:CodeBook{T,D}} = D
ndims(cb::CodeBook) = ndims(typeof(cb))

decimations(fb::FilterBank) = fb.decimationFactor
nchannels(fb::FilterBank) = sum(fb.nChannels)
orders(fb::FilterBank) = fb.polyphaseOrder

kernels(fb::FilterBank) = fb.kernels
kernelsize(fb::FilterBank) = decimations(fb) .* (1 .+ orders(fb))

struct Rnsolt{T,D} <: AbstractNsolt{T,D}
    decimationFactor::NTuple{D, Int}
    polyphaseOrder::NTuple{D, Int}
    nChannels::Tuple{Int,Int}

    initMatrices::Array{AbstractMatrix{T},1}
    propMatrices::Array{Array{AbstractMatrix{T},1},1}

    matrixC::Matrix

    # Constructors
    Rnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Union{Tuple{Int,Int}, Integer}; kwargs...) where {D} = Rnsolt(Float64, df, ppo, nChs; kwargs...)

    Rnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Integer) where {D,T} = Rnsolt(T, df, ppo, (cld(nChs,2), fld(nChs,2)))

    function Rnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Tuple{Int,Int}) where {T,D}
        mts = get_rnsolt_default_matrices(Val(nChs[1]==nChs[2]), T, ppo, nChs)
        Rnsolt(df, ppo, nChs, mts...)
    end

    function Rnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Tuple{Int,Int}, initMts::Vector{MT}, propMts::Vector{Vector{MT}}) where {T,D,MT<:AbstractArray{T}}
        P = sum(nChs)
        M = prod(df)
        if !(cld(M,2) <= nChs[1] <= P - fld(M,2)) || !(fld(M,2) <= nChs[2] <= P - cld(M,2))
            throw(ArgumentError("Invalid number of channels. "))
        end
        if nChs[1] != nChs[2] && any(isodd.(ppo))
            throw(ArgumentError("Sorry, odd-order Type-II CNSOLT hasn't implemented yet. received values: decimationFactor=$df, nChannels=$nChs, polyphaseOrder = $ppo"))
        end

        TC = if T <: AbstractFloat; T else Float64 end
        mtxc = reverse(permdctmtx(TC, df...); dims=2)

        new{T,D}(df, ppo, nChs, initMts, propMts, mtxc)
    end

    Rnsolt(df::Integer, ppo::Integer, nChs; kwargs...) = Rnsolt(Float64, df, ppo, nChs; kwargs...)
    Rnsolt(::Type{T}, df::Integer, ppo::Integer, nChs::Integer; kwargs...) where {T} = Rnsolt(T, df, ppo, (cld(nChs,2),fld(nChs,2)); kwargs...)
    function Rnsolt(::Type{T}, df::Integer, ppo::Integer, nChs::Tuple{Int,Int}; dims::Integer=1) where {T}
        Rnsolt(T, (fill(df,dims)...,), (fill(ppo,dims)...,), nChs)
    end
end

promote_rule(::Type{Rnsolt{TA,D}}, ::Type{Rnsolt{TB,D}}) where {D,TA,TB} = Rnsolt{promote_type(TA,TB),D}

similar(nsolt::Rnsolt{T,DS}, element_type::Type=T, df::NTuple{DD}=nsolt.decimationFactor, ord::NTuple{DD}=nsolt.polyphaseOrder, nch::Union{Integer,Tuple{Int,Int}}=nsolt.nChannels) where {T,DS,DD} = Rnsolt(element_type, df, ord, nch)

istype1(nsolt::Rnsolt) = nsolt.nChannels[1] == nsolt.nChannels[2]

Rnsolt1D{T} = Rnsolt{T,1}
Rnsolt2D{T} = Rnsolt{T,2}
Rnsolt3D{T} = Rnsolt{T,3}

struct Cnsolt{T,D} <: AbstractNsolt{T,D}
    decimationFactor::NTuple{D, Int}
    polyphaseOrder::NTuple{D, Int}
    nChannels::Int

    initMatrices::Vector{AbstractMatrix{T}}
    propMatrices::Vector{Vector{AbstractMatrix{T}}}
    paramAngles::Vector{Vector{AbstractVector{T}}}
    symmetry::Diagonal
    matrixF::Matrix

    Cnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Int; kwargs...) where {D} = Cnsolt(Float64, df, ppo, nChs; kwargs...)

    function Cnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Int; kwargs...) where {T,D}
        mts = get_cnsolt_default_matrices(Val(iseven(nChs)), T, ppo, nChs)
        paramAngs = Vector{Vector{T}}[ [ zeros(fld(nChs,4)) for n in 1:ppo[pd] ] for pd in 1:D ]

        Cnsolt(df, ppo, nChs, mts..., paramAngs; kwargs...)
    end

    function Cnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Integer, initMts::Vector{MT}, propMts::Vector{Vector{MT}}, paramAngs::Vector{Vector{VT}}; symmetry::AbstractVector=ones(nChs), matrixF::AbstractMatrix=reverse(cdftmtx(TF, df...); dims=2)) where {T,D,MT<:AbstractMatrix{T},VT<:AbstractVector{T}}
        if prod(df) > nChs
            throw(ArgumentError("The number of channels must be equal or greater than a product of the decimation factor."))
        end

        TF = if T <: AbstractFloat; T else Float64 end
        sym = Diagonal{Complex{TF}}(symmetry)

        new{T,D}(df, ppo, nChs, initMts, propMts, paramAngs, sym, matrixF)
    end

    Cnsolt(df::Integer, ppo::Integer, nChs::Integer; kwargs...) = Cnsolt(Float64, df, ppo, nChs; kwargs...)
    function Cnsolt(::Type{T}, df::Integer, ppo::Integer, nChs::Integer; dims::Integer=1) where {T}
        Cnsolt(T, (fill(df,dims)...,), (fill(ppo,dims)...,), nChs)
    end

    Cnsolt(nsolt::Rnsolt) = lifting(Val(istype1(nsolt)), nsolt)
end

promote_rule(::Type{Cnsolt{TA,D}}, ::Type{Cnsolt{TB,D}}) where {D,TA,TB} = Cnsolt{promote_type(TA,TB),D}

similar(nsolt::Cnsolt{T,DS}, element_type::Type=T, df::NTuple{DD}=nsolt.decimationFactor, ord::NTuple{DD}=nsolt.polyphaseOrder, nch::Integer=nsolt.nChannels) where {T,DS,DD} = Cnsolt(element_type, df, ord, nch)

istype1(nsolt::Cnsolt) = iseven(nsolt.nChannels)

istype2(nsolt::AbstractNsolt) = !istype1(nsolt)

Cnsolt1D{T} = Cnsolt{T,1}
Cnsolt2D{T} = Cnsolt{T,2}
Cnsolt3D{T} = Cnsolt{T,3}

TypeI = Val{true}
TypeII = Val{false}

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

include("sparseCoding.jl")

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
