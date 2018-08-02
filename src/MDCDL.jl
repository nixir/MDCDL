module MDCDL # Multi-Dimensional Convolutional Dictionary Learning

using LinearAlgebra

import Base.promote_rule

include("basicComplexDSP.jl")

export PolyphaseVector
export FilterBank, PolyphaseFB, ParallelFB, Cnsolt, Rnsolt
export Multiscale, MultiLayerCsc
export analyze, analyze!, synthesize, sinthesize!, adjoint_synthesize
# export cconv
export upsample, downsample
export atmimshow
# export getAnalysisBank
export getAnalysisFilters, getSynthesisFilters
export getAngleParameters, setAngleParameters!
export mdarray2polyphase, polyphase2mdarray
export iht


struct PolyphaseVector{T,D} <: AbstractArray{T,2}
    data::Matrix{T}
    nBlocks::NTuple{D, Int}
end

promote_rule(::Type{PolyphaseVector{TA,D}}, ::Type{PolyphaseVector{TB,D}}) where {TA,TB,D} = PolyphaseVector{promote_type(TA,TB), D}

abstract type CodeBook{T,D} end
abstract type FilterBank{T,D} <: CodeBook{T,D} end
abstract type PolyphaseFB{T,D} <: FilterBank{T,D} end

struct Rnsolt{T,D,S} <: PolyphaseFB{T,D}
    decimationFactor::NTuple{D, Int}
    polyphaseOrder::NTuple{D, Int}
    nChannels::Tuple{Int,Int}

    initMatrices::Array{Matrix{T},1}
    propMatrices::Array{Array{Matrix{T},1},1}

    matrixC::Matrix{T}

    function Rnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Tuple{Int, Int}) where {T,D}
        P = sum(nChs)
        M = prod(df)
        if !(cld(M,2) <= nChs[1] <= P - fld(M,2)) || !(fld(M,2) <= nChs[2] <= P - cld(M,2))
            throw(ArgumentError("Invalid number of channels. "))
        end
        if nChs[1] != nChs[2] && any(isodd.(ppo))
            throw(ArgumentError("Sorry, odd-order Type-II CNSOLT hasn't implemented yet. received values: decimationFactor=$df, nChannels=$nChs, polyphaseOrder = $ppo"))
        end

        if nChs[1] == nChs[2]
            S = :TypeI
            initMts = Array[ Matrix{T}(I, p, p) for p in nChs ]
            propMts = Array[
                Array[
                    (iseven(n) ? 1 : -1) * Matrix{T}(I, nChs[1], nChs[1])
                for n in 1:ppo[pd] ]
            for pd in 1:D ]
        else
            S = :TypeII
            initMts = Array[ Matrix{T}(I, p, p) for p in nChs ]
            propMts = if nChs[1] > nChs[2]
                [
                    vcat(
                        fill(Array[-Matrix{T}(I,nChs[2],nChs[2]), Matrix{T}(I,nChs[1],nChs[2]) ], fld(ppo[pd],2))...
                    )
                for pd in 1:D]
            else
                [
                    vcat(
                        fill(Array[ Matrix{T}(I,nChs[1],nChs[1]), -Matrix{T}(I,nChs[2],nChs[2]) ], fld(ppo[pd],2))...
                    )
                for pd in 1:D]
            end
        end

        mtxc = reverse(MDCDL.permdctmtx(T, df...); dims=2)

        new{T,D,S}(df, ppo, nChs, initMts, propMts, mtxc)
    end
    Rnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Tuple{Int,Int}; kwargs...) where {D} = Rnsolt(Float64, df, ppo, nChs; kwargs...)
end

promote_rule(::Type{Rnsolt{TA,D,S}}, ::Type{Rnsolt{TB,D,S}}) where {D,S,TA,TB} = Rnsolt{promote_type(TA,TB),D,S}

struct Cnsolt{T,D,S} <: PolyphaseFB{Complex{T},D}
    decimationFactor::NTuple{D, Int}
    polyphaseOrder::NTuple{D, Int}
    nChannels::Int

    initMatrices::Vector{Matrix{T}}
    propMatrices::Vector{Vector{Matrix{T}}}
    paramAngles::Vector{Vector{Vector{T}}}
    symmetry::Diagonal{Complex{T}}
    matrixF::Matrix{Complex{T}}

    # constructor
    function Cnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Int) where {T,D}
        if prod(df) > nChs
            throw(ArgumentError("The number of channels must be equal or greater than a product of the decimation factor."))
        end

        if iseven(nChs)
            S = :TypeI
            initMts = Array[ Matrix{T}(I,nChs,nChs) ]
            propMts = Array[
                Array[
                    (iseven(n) ? -1 : 1) * Matrix{T}(I,fld(nChs,2),fld(nChs,2))
                for n in 1:2*ppo[pd] ]
            for pd in 1:D ]
        else
            if any(isodd.(ppo))
                throw(ArgumentError("Sorry, odd-order Type-II CNSOLT hasn't implemented yet."))
            end
            cch = cld(nChs, 2)
            fch = fld(nChs, 2)
            S = :TypeII
            initMts = Array[ Matrix{T}(I, nChs, nChs) ]
            propMts = [
                vcat(fill(Array[
                    Matrix{T}(I,fch,fch), -Matrix{T}(I,fch,fch), Matrix{T}(I,cch,cch), Matrix(Diagonal(vcat(fill(T(-1), fld(nChs,2))..., T(1))))
                ], fld(ppo[pd],2))...)
            for pd in 1:D]
        end
        paramAngs = Array[
            Array[ zeros(T,fld(nChs,4)) for n in 1:ppo[pd] ]
        for pd in 1:D ]
        sym = Diagonal{Complex{T}}(ones(nChs))
        mtxf = reverse(MDCDL.cdftmtx(T, df...); dims=2)

        new{T,D,S}(df, ppo, nChs, initMts, propMts, paramAngs, sym, mtxf)
    end
    Cnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Int; kwargs...) where {D} = Cnsolt(Float64, df, ppo, nChs; kwargs...)
end

promote_rule(::Type{Cnsolt{TA,D,S}}, ::Type{Cnsolt{TB,D,S}}) where {D,S,TA,TB} = Cnsolt{promote_type(TA,TB),D,S}

struct ParallelFB{T,D} <: FilterBank{T,D}
    decimationFactor::NTuple{D, Int}
    polyphaseOrder::NTuple{D, Int}
    nChannels::Int

    analysisFilters::Vector{Array{T,D}}
    synthesisFilters::Vector{Array{T,D}}

    function ParallelFB(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Int) where {T,D}
        szFilters = df .* (ppo .+ 1)
        afs = [ Array{T, D}(szFilters...) for p in 1:nChs ]
        sfs = [ Array{T, D}(szFilters...) for p in 1:nChs ]
        new{T, D}(df, ppo, nChs, afs, sfs)
    end

    function ParallelFB(fb::PolyphaseFB{T,D}) where {T,D}
        afs = getAnalysisFilters(fb)
        fsf = getSynthesisFilters(fb)
        new{T, D}(fb.decimationFactor, fb.polyphaseOrder, sum(fb.nChannels), afs, fsf)
    end
end

promote_rule(::Type{ParallelFB{TA,D}}, ::Type{ParallelFB{TB,D}}) where {TA,TB,D} = ParallelFB{promote_type(TA,TB),D}

struct Multiscale{T,D} <: CodeBook{T,D}
    filterBank::FilterBank{T,D}
    treeLevel::Int
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

include("sparseCoding.jl")

include("blockproc.jl")
include("orthonormalMatrixSystem.jl")
include("polyphaseMatrices.jl")

include("analysisSystem.jl")
include("synthesisSystem.jl")

include("parameterizeFilterBanks.jl")

include("dictionaryLearning.jl")
include("mlcsc.jl")

include("view.jl")
include("io.jl")

end
