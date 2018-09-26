module MDCDL # Multi-Dimensional Convolutional Dictionary Learning

using LinearAlgebra
using ImageFiltering
using ImageFiltering.Algorithm: Alg, FIR, FFT
using ComputationalResources: AbstractResource, CPU1

import Base: promote_rule, eltype, ndims, similar
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
export kernels, analysiskernels, synthesiskernels, kernelsize
export getrotations, setrotations!
export mdarray2polyphase, polyphase2mdarray
export iht
export AbstractOperator
export NsoltOperator, ConvolutionalOperator
export createOperator, createAnalyzer, createSynthesizer
export Shapes, Optimizers, SparseCoders

module Shapes
    abstract type AbstractShape end
    struct Default <: AbstractShape end
    struct Vec <: AbstractShape end
    struct Augumented <: AbstractShape end
end

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
        if nChs[1] == nChs[2]   # Type-I R-NSOLT
            initMts = Matrix{T}[ Matrix(I, p, p) for p in nChs ]
            propMts = Vector{Matrix{T}}[
                [
                    (iseven(n) ? 1 : -1) .* Matrix(I, nChs[1], nChs[1])
                for n in 1:ppo[pd] ]
            for pd in 1:D ]
        else                    # Type-II R-NSOLT
            initMts = Matrix{T}[ Matrix(I, p, p) for p in nChs ]
            chx, chn = maximum(nChs), minimum(nChs)
            propMts = Vector{Matrix{T}}[
                vcat(
                    fill([ -Matrix(I, chn, chn), Matrix(I, chx, chx) ], fld(ppo[pd],2))...
                )
            for pd in 1:D ]
        end

        Rnsolt(df, ppo, nChs, initMts, propMts)
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

    function Cnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Int) where {T,D}
        if iseven(nChs) # Type-I C-NSOLT
            initMts = Matrix{T}[ Matrix(I,nChs,nChs) ]
            propMts = Vector{Matrix{T}}[
                [
                    (iseven(n) ? -1 : 1) * Matrix(I,fld(nChs,2),fld(nChs,2))
                for n in 1:2*ppo[pd] ]
            for pd in 1:D ]
        else            # Type-II C-NSOLT
            if any(isodd.(ppo))
                throw(ArgumentError("Sorry, odd-order Type-II CNSOLT hasn't implemented yet."))
            end
            cch = cld(nChs, 2)
            fch = fld(nChs, 2)
            initMts = Matrix{T}[ Matrix(I, nChs, nChs) ]
            propMts = Vector{Matrix{T}}[
                vcat(fill([
                    Matrix(I,fch,fch), -Matrix(I,fch,fch), Matrix(I,cch,cch), Matrix(Diagonal(vcat(fill(-1, fld(nChs,2))..., 1)))
                ], fld(ppo[pd],2))...)
            for pd in 1:D]
        end
        paramAngs = Vector{Vector{T}}[ [ zeros(fld(nChs,4)) for n in 1:ppo[pd] ] for pd in 1:D ]

        Cnsolt(df, ppo, nChs, initMts, propMts, paramAngs)
    end

    function Cnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Int, initMts::Vector{MT}, propMts::Vector{Vector{MT}}, paramAngs::Vector{Vector{VT}}) where {T,D,MT<:AbstractMatrix{T},VT<:AbstractVector{T}}
        if prod(df) > nChs
            throw(ArgumentError("The number of channels must be equal or greater than a product of the decimation factor."))
        end

        TF = if T <: AbstractFloat; T else Float64 end
        sym = Diagonal{Complex{TF}}(ones(nChs))
        mtxf = reverse(cdftmtx(TF, df...); dims=2)

        new{T,D}(df, ppo, nChs, initMts, propMts, paramAngs, sym, mtxf)
    end
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

    kernels::Vector{AbstractArray{T,D}}

    function ParallelFilters(ker::Vector{A}, df::NTuple{D,Int}, ord::NTuple{D,Int}, nch::Integer) where {T,D,A<:AbstractArray{T,D}}
        new{T,D}(df, ord, nch, ker)
    end

    function ParallelFilters(::Type{T}, df::NTuple{D,Int}, ord::NTuple{D,Int}, nch::Integer) where {T,D}
        new{T,D}(df, ord, nch, fill(zeros(df .* (ord .+ 1)), nch))
    end
end

struct MultiLayerCsc{T,D} <: CodeBook{T,D}
    nLayers::Int

    dictionaries::Vector
    # lambdas::Vector{T}
    # proxOperators::Vector{Function}

    function MultiLayerCsc{T,D}(nl::Integer) where{T,D}
        dics = Vector(nl)
        # lmds = fill(convert(T,1.0), (nl,))
        # pos = fill( (y,lm) -> identity(y) , (nl,))

        # new{T,D}(nl, dics, lmds, pos)
        new{T,D}(nl, dics)
    end
end

get_outputsize(s::Shapes.AbstractShape, pfb, insz::NTuple) = get_outputsize(s, fld.(insz, decimations(pfb)), insz, nchannels(pfb))

get_outputsize(::Shapes.Default, dcsz::NTuple, insz::NTuple, nch::Integer) = (nch, dcsz...)
get_outputsize(::Shapes.Augumented, dcsz::NTuple, insz::NTuple, nch::Integer) = (dcsz..., nch)
get_outputsize(::Shapes.Vec, dcsz::NTuple, insz::NTuple, nch::Integer) = (prod(dcsz) * nch,)

abstract type AbstractOperator{T,D} end

createAnalyzer(obj, args...; kwargs...) = createOperator(obj, args...; kwargs...)
createSynthesizer(obj, args...; kwargs...) = createOperator(obj, args...; kwargs...)
createAnalyzer(::Type{OP}, obj, args...; kwargs...) where {OP<:AbstractOperator} = OP(obj, args...; kwargs...)
createSynthesizer(::Type{OP}, obj, args...; kwargs...) where {OP<:AbstractOperator} = OP(obj, args...; kwargs...)
createAnalyzer(::Type{AbstractOperator}, obj, args...; kwargs...) = createAnalyzer(obj, args...; kwargs...)
createSynthesizer(::Type{AbstractOperator}, obj, args...; kwargs...) = createSynthesizer(obj, args...; kwargs...)

struct NsoltOperator{T,D} <: AbstractOperator{T,D}
    shape::Shapes.AbstractShape
    insize::NTuple
    outsize::NTuple

    nsolt::AbstractNsolt{T,D}
    border::Symbol

    function NsoltOperator(ns::AbstractNsolt{T,D}, insz::NTuple, outsz::NTuple; shape=Shapes.Default(), border=:circular) where {T,D}
        new{T,D}(shape, insz, outsz, ns, border)
    end

    function NsoltOperator(ns::AbstractNsolt{T,D}, insz::NTuple; shape=Shapes.Default(), kwargs...) where {T,D}
        outsz = get_outputsize(shape, ns, insz)
        NsoltOperator(ns, insz, outsz; shape=shape, kwargs...)
    end

    function NsoltOperator(ns::AbstractNsolt, x::AbstractArray; kwargs...)
        NsoltOperator(ns, size(x); kwargs...)
    end
end

decimations(nsop::NsoltOperator) = decimations(nsop.nsolt)
orders(nsop::NsoltOperator) = orders(nsop.nsolt)
nchannels(nsop::NsoltOperator) = nchannels(nsop.nsolt)

createOperator(ns::AbstractNsolt, x; kwargs...) = NsoltOperator(ns, x; kwargs...)

struct ConvolutionalOperator{T,D} <: AbstractOperator{T,D}
    insize::NTuple
    outsize::NTuple
    shape::Shapes.AbstractShape

    parallelFilters::ParallelFilters{T,D}

    border::Symbol
    resource::AbstractResource

    function ConvolutionalOperator(filters::ParallelFilters{T,D}, insz::NTuple, outsz::NTuple; shape=Shapes.Default(), border=:circular, resource=CPU1(FIR())) where {T,D}
        new{T,D}(insz, outsz, shape, filters, border, resource)
    end

    function ConvolutionalOperator(pfs::ParallelFilters{T,D}, insz::NTuple{D,Int}; shape=Shapes.Default(), kwargs...) where {T,D}
        outsz = get_outputsize(shape, pfs, insz)
        ConvolutionalOperator(pfs, insz, outsz; shape=shape, kwargs...)
    end

    function ConvolutionalOperator(kernel::AbstractArray{AR}, insz::NTuple, df::NTuple{D}, ord::NTuple{D}, nch::Integer; kwargs...) where {T,D,AR<:AbstractArray{T,D}}
        ConvolutionalOperator(ParallelFilters(kernel, df, ord, nch), insz; kwargs...)
    end
end

decimations(co::ConvolutionalOperator) = decimations(co.parallelFilters)
orders(co::ConvolutionalOperator) = orders(co.parallelFilters)
nchannels(co::ConvolutionalOperator) = nchannels(co.parallelFilters)

function createAnalyzer(::Type{CO}, pfb::PolyphaseFB, insz::NTuple; kwargs...) where {CO<:ConvolutionalOperator}
    CO(analysiskernels(pfb), insz, decimations(pfb), orders(pfb), nchannels(pfb); kwargs...)
end

function createSynthesizer(::Type{CO}, pfb::PolyphaseFB, insz::NTuple; kwargs...) where {CO<:ConvolutionalOperator}
    CO(synthesiskernels(pfb), insz, decimations(pfb), orders(pfb), nchannels(pfb); kwargs...)
end

struct MultiscaleOperator{T,D} <: AbstractOperator{T,D}
    insize::NTuple{D,T}
    shape::Shapes.AbstractShape

    operators::Vector{AbstractOperator{T,D}}

    function MultiscaleOperator(ops::Vector{X}, sz::NTuple{D,Int}; shape=Shapes.Default()) where {T,D,X<:AbstractOperator{T,D}}
        new{T,D}(sz, shape, ops)
    end
end

nchannels(msop::MultiscaleOperator) = nchannels.(msop.operators)

function createAnalyzer(cbs::NTuple{N,CB}, sz::NTuple; shape=Shapes.Default(), kwargs...) where {N,CB<:CodeBook}
    szxs = [ fld.(sz, decimations(cbs[lv]).^(lv-1)) for lv in 1:N ]
    ops = map((acb, asz)->createAnalyzer(acb, asz; shape=shape, kwargs...), cbs, szxs)
    MultiscaleOperator(ops, sz; shape=shape)
end

function createSynthesizer(cbs::NTuple{N,CB}, sz::NTuple; shape=Shapes.Default(), kwargs...) where {N,CB<:CodeBook}
    szxs = [ fld.(sz, decimations(cbs[lv]).^(lv-1)) for lv in 1:N ]
    ops = map((acb, asz)->createSynthesizer(acb, asz; shape=shape, kwargs...), cbs, szxs)
    MultiscaleOperator(ops, sz; shape=shape)
end

include("sparseCoding.jl")

include("orthonormalMatrixSystem.jl")
include("polyphaseMatrices.jl")

include("analysisSystem.jl")
include("synthesisSystem.jl")
include("transforms.jl")

include("parameterizeFilterBanks.jl")

include("dictionaryLearning.jl")
include("mlcsc.jl")

include("recipes.jl")
include("io.jl")

include("utils.jl")

end
