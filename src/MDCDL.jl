module MDCDL # Multi-Dimensional Convolutional Dictionary Learning

using LinearAlgebra
using ImageFiltering
using ImageFiltering.Algorithm: Alg, FIR, FFT
using ComputationalResources: AbstractResource, CPU1

import Base: promote_rule, eltype, ndims, similar
import Random: rand!

include("basicComplexDSP.jl")

export PolyphaseVector
export FilterBank, PolyphaseFB, ParallelFB, AbstractNsolt, Cnsolt, Rnsolt
export Multiscale, MultiLayerCsc
export analyze, synthesize, adjoint_synthesize
export upsample, downsample
export serialize, deserialize
export analysisbank
export kernels, analysiskernels, synthesiskernels
export getrotations, setrotations!
export mdarray2polyphase, polyphase2mdarray
export iht
# export Analyzer, VecAnalyzer
# export Synthesizer, VecSynthesizer
export AbstractOperator
export NsoltOperator
export ConvolutionalOperator
export createAnalyzer, createSynthesizer
export createMultiscaleAnalyzer, createMultiscaleSynthesizer
export Shapes

module Shapes
    abstract type Shape end
    struct Default <: Shape end
    struct Vec <: Shape end
    struct Augumented <: Shape end
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

struct Rnsolt{T,D} <: AbstractNsolt{T,D}
    category::Symbol
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

        categ = if nChs[1] == nChs[2]; :TypeI else :TypeII end

        TC = if T <: AbstractFloat; T else Float64 end
        mtxc = reverse(permdctmtx(TC, df...); dims=2)

        new{T,D}(categ, df, ppo, nChs, initMts, propMts, mtxc)
    end
end

promote_rule(::Type{Rnsolt{TA,D}}, ::Type{Rnsolt{TB,D}}) where {D,TA,TB} = Rnsolt{promote_type(TA,TB),D}

similar(nsolt::Rnsolt{T,DS}, element_type::Type=T, df::NTuple{DD}=nsolt.decimationFactor, ord::NTuple{DD}=nsolt.polyphaseOrder, nch::Union{Integer,Tuple{Int,Int}}=nsolt.nChannels) where {T,DS,DD} = Rnsolt(element_type, df, ord, nch)

struct Cnsolt{T,D} <: AbstractNsolt{T,D}
    category::Symbol
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

        categ = if iseven(nChs); :TypeI else :TypeII end
        TF = if T <: AbstractFloat; T else Float64 end
        sym = Diagonal{Complex{TF}}(ones(nChs))
        mtxf = reverse(cdftmtx(TF, df...); dims=2)

        new{T,D}(categ, df, ppo, nChs, initMts, propMts, paramAngs, sym, mtxf)
    end
end

promote_rule(::Type{Cnsolt{TA,D}}, ::Type{Cnsolt{TB,D}}) where {D,TA,TB} = Cnsolt{promote_type(TA,TB),D}

similar(nsolt::Cnsolt{T,DS}, element_type::Type=T, df::NTuple{DD}=nsolt.decimationFactor, ord::NTuple{DD}=nsolt.polyphaseOrder, nch::Integer=nsolt.nChannels) where {T,DS,DD} = Cnsolt(element_type, df, ord, nch)

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

abstract type AbstractOperator{T,D} end

struct NsoltOperator{T,D} <: AbstractOperator{T,D}
    shape::Shapes.Shape
    insize::NTuple
    outsize::NTuple

    nsolt::AbstractNsolt{T,D}
    border::Symbol

    function NsoltOperator(ns::AbstractNsolt{T,D}, insz::NTuple, outsz::NTuple; shape=Shapes.Default(), border=:circular) where {T,D}
        new{T,D}(shape, insz, outsz, ns, border)
    end

    function NsoltOperator(ns::AbstractNsolt{T,D}, insz::NTuple; shape=Shapes.Default(), kwargs...) where {T,D}
        # new{T,D}(mode, shape, insz, outsz, ns, border)
        dcsz = fld.(insz, ns.decimationFactor)
        outsz = if shape isa Shapes.Default
            (sum(ns.nChannels), dcsz...,)
        elseif shape isa Shapes.Augumented
            (dcsz..., sum(ns.nChannels),)
        elseif shape isa Shapes.Vec
            (prod(dcsz) * sum(ns.nChannels),)
        else
            error("Invalid augument")
        end
        NsoltOperator(ns, insz, outsz; shape=shape, kwargs...)
    end

    function NsoltOperator(ns::AbstractNsolt, x::AbstractArray; kwargs...)
        NsoltOperator(ns, size(x); kwargs...)
    end
end

createAnalyzer(ns::AbstractNsolt, args...; kwargs...) = NsoltOperator(ns, args...; kwargs...)
createSynthesizer(ns::AbstractNsolt, args...; kwargs...) = NsoltOperator(ns, args...; kwargs...)

struct ConvolutionalOperator{T,D} <: AbstractOperator{T,D}
    insize::NTuple
    outsize::NTuple
    shape::Shapes.Shape

    kernels::Vector{Array{T,D}}

    decimationFactor::NTuple{D,Int}
    polyphaseOrder::NTuple{D,Int}
    nChannels::Int

    border::Symbol
    resource::AbstractResource

    function ConvolutionalOperator(kernels::Vector{Array{T,D}}, insz::NTuple, outsz::NTuple, df::NTuple{D,Int}, ord::NTuple{D,Int}, nch::Int; shape=Shapes.Default(), border=:circular, resource=CPU1(FIR())) where {T,D}
        new{T,D}(insz, outsz, shape, kernels, df, ord, nch, border, resource)
    end

    function ConvolutionalOperator(kernels::Vector{Array{T,D}}, insz::NTuple{D,Int}, df::NTuple{D,Int}, ord::NTuple{D,Int}, nch::Int; shape=Shapes.Default(), kwargs...) where {T,D}
        dcsz = fld.(insz, df)
        outsz = if shape isa Shapes.Default
            (sum(nch), dcsz...,)
        elseif shape isa Shapes.Augumented
            (dcsz..., nch,)
        elseif shape isa Shapes.Vec
            (prod(dcsz) * nch,)
        else
            error("Invalid augument")
        end
        ConvolutionalOperator(kernels, insz, outsz, df, ord, nch; shape=shape, kwargs...)
    end

    function ConvolutionalOperator(kernels::Vector{Array{T,D}}, sz::NTuple{D,Int}; decimation::NTuple{D,Int}, kwargs...) where {T,D}
        nch = length(kernels)
        szFilter = size(kernels[1])
        # if any(map(ker->size(ker) != szFilter, kernels))
        #     error("size mismatch")
        # end
        ord = fld.(szFilter, decimation) .- 1
        ConvolutionalOperator(kernels, sz, decimation, ord, nch)
    end

    function ConvolutionalOperator(pfb::PolyphaseFB{T,D}, sz::NTuple{D,Int}, mode::Symbol; kwargs...) where {T,D}
        afs = if mode == :analyzer
            analysiskernels(pfb)
        elseif mode == :synthesizer
            synthesiskernels(pfb)
        end
        ConvolutionalOperator(afs, sz, pfb.decimationFactor, pfb.polyphaseOrder, sum(pfb.nChannels); kwargs...)
    end

    function ConvolutionalOperator(pfb::PolyphaseFB, x::AbstractArray, args...; kwargs...)
        ConvolutionalOperator(pfb, size(x), args...; kwargs...)
    end
end

createAnalyzer(ker::Vector{Array{T,D}}, args...; kwargs...) where {T,D} = ConvolutionalOperator(ker, args...; kwargs...)
createSynthesizer(ker::Vector{Array{T,D}}, args...; kwargs...) where {T,D} = ConvolutionalOperator(ker, args...; kwargs...)

struct MultiscaleOperator{T,D} <: AbstractOperator{T,D}
    insize::NTuple{D,T}
    shape::Shapes.Shape

    operators::Vector{AbstractOperator{T,D}}

    function MultiscaleOperator(ops::Vector{X}, sz::NTuple{D,Int}; shape=Shapes.Default()) where {T,D,X<:AbstractOperator{T,D}}
        new{T,D}(sz, shape, ops)
    end
end

function createMultiscaleAnalyzer(ns::AbstractNsolt{T,D}, sz::NTuple{D,Int}; level, shape=Shapes.Default(), kwargs...) where {T,D}
    szxs = [ fld.(sz, ns.decimationFactor.^(lv-1)) for lv in 1:level ]
    ops = map(t->createAnalyzer(ns, t; shape=Shapes.Default(), kwargs...), szxs)
    MultiscaleOperator(ops, sz; shape=shape)
end

function createMultiscaleSynthesizer(ns::AbstractNsolt{T,D}, sz::NTuple{D,Int}; level, shape=Shapes.Default(), kwargs...) where {T,D}
    szxs = [ fld.(sz, ns.decimationFactor.^(lv-1)) for lv in 1:level ]
    ops = map(t->createSynthesizer(ns, t; shape=shape, kwargs...), szxs)
    MultiscaleOperator(ops, sz; shape=shape)
end

include("sparseCoding.jl")

include("orthonormalMatrixSystem.jl")
include("polyphaseMatrices.jl")

include("analysisSystem.jl")
include("synthesisSystem.jl")

include("parameterizeFilterBanks.jl")

include("dictionaryLearning.jl")
include("mlcsc.jl")

include("recipes.jl")
include("io.jl")

include("utils.jl")

end
