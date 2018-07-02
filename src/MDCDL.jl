module MDCDL # Multi-Dimensional Convolutional Dictionary Learning

import Base.promote_rule

include("basicComplexDSP.jl")

export PolyphaseVector
export FilterBank, PolyphaseFB, ParallelFB, Cnsolt, Rnsolt
export MultiLayerCsc
export analyze, synthesize
# export cconv
export upsample, downsample
export atmimshow
# export getAnalysisBank
export getAnalysisFilters, getSynthesisFilters
export mdfilter
export getAngleParameters, setAngleParameters!


struct PolyphaseVector{T,D} <: AbstractArray{T,2}
    data::Matrix{T}
    nBlocks::NTuple{D, Int}
end

promote_rule(::Type{PolyphaseVector{TA,D}}, ::Type{PolyphaseVector{TB,D}}) where {TA,TB,D} = PolyphaseVector{promote_type(TA,TB), D}

abstract type FilterBank{T,D} end
abstract type PolyphaseFB{T,D} <: FilterBank{T,D} end

struct Rnsolt{D,S,T} <: PolyphaseFB{T,D}
    decimationFactor::NTuple{D, Int}
    nChannels::Tuple{Int,Int}
    polyphaseOrder::NTuple{D, Int}

    nVanishingMoment::Int

    initMatrices::Array{Matrix{T},1}
    propMatrices::Array{Array{Matrix{T},1},1}

    matrixC::Matrix{T}

    function Rnsolt(df::NTuple{D,Int}, nChs::Tuple{Int, Int}, ppo::NTuple{D,Int}; vanishingMoment::Int = 0, dataType = Float64) where D
        T = dataType
        P = sum(nChs)
        M = prod(df)
        if !(cld(M,2) <= nChs[1] <= P - fld(M,2)) || !(fld(M,2) <= nChs[2] <= P - cld(M,2))
            throw(ArgumentError("Invalid number of channels. "))
        end
        if vanishingMoment != 0 && vanishingMoment != 1
            throw(ArgumentError("The number of vanishing moments must be 0 or 1."))
        end

        if nChs[1] == nChs[2]
            S = 1
            initMts = Array[ eye(T, p) for p in nChs ]
            propMts = Array[ Array[ (n % 2 == 0 ? 1 : -1) * eye(T, nChs[1]) for n in 1:ppo[pd] ] for pd in 1:D ] # Consider only Type-I NSOLT
        else
            S = 2
            initMts = Array[ eye(T, p) for p in nChs ]
            propMts = if nChs[1] > nChs[2]
                [ vcat(fill(Array[ -eye(T,nChs[2]), eye(T,nChs[1]) ], ppo[pd])...) for pd in 1:D]
            else
                [ vcat(fill(Array[ eye(T,nChs[1]), -eye(T,nChs[2]) ], ppo[pd])...) for pd in 1:D]
            end
        end

        mtxc = flipdim(MDCDL.permdctmtx(df...),2)

        new{D,S,T}(df, nChs, ppo, vanishingMoment, initMts, propMts, mtxc)
    end
end

promote_rule(::Type{Rnsolt{D,S,TA}}, ::Type{Rnsolt{D,S,TB}}) where {D,S,TA,TB} = Rnsolt{D,S,promote_type(TA,TB)}

struct Cnsolt{D,S,T} <: PolyphaseFB{Complex{T},D}
    decimationFactor::NTuple{D, Int}
    nChannels::Int
    polyphaseOrder::NTuple{D, Int}
    # directionPermutation::NTuple{D, Int}

    initMatrices::Vector{Matrix{T}}
    propMatrices::Vector{Vector{Matrix{T}}}
    paramAngles::Vector{Vector{Vector{T}}}
    symmetry::Diagonal{Complex{T}}
    matrixF::Matrix{Complex{T}}

    # constructor
    function Cnsolt(df::NTuple{D,Int}, nChs::Int, ppo::NTuple{D,Int}; vanishingMoment::Int = 0, dataType = Float64) where D
        T = dataType
        if prod(df) > nChs
            throw(ArgumentError("The number of channels must be equal or greater than a product of the decimation factor."))
        end

        if nChs % 2 == 0
            S = 1
            initMts = Array[ eye(T,nChs) ]
            propMts = Array[ Array[ (n % 2 == 0 ? -1 : 1) * eye(T,fld(nChs,2)) for n in 1:2*ppo[pd] ] for pd in 1:D ] # Consider only Type-I CNSOLT
        else
            if any(ppo .% 2 .!= 0)
                throw(ArgumentError("Sorry, odd-order Type-II CNSOLT hasn't implemented yet."))
            end
            cch = cld(nChs, 2)
            fch = fld(nChs, 2)
            S = 2
            initMts = Array[ eye(T, nChs) ]
            propMts = [ vcat(fill(Array[ eye(T,fch), -eye(T,fch), eye(T,cch), diagm(vcat(fill(T(-1), fld(nChs,2))..., T(1))) ], ppo[pd])...) for pd in 1:D]
        end
        paramAngs = Array[ Array[ zeros(T,fld(nChs,4)) for n in 1:ppo[pd] ] for pd in 1:D ]
        sym = Diagonal{Complex{T}}(ones(nChs))
        mtxf = flipdim(MDCDL.cdftmtx(df...),2)

        new{D,S,T}(df, nChs, ppo, initMts, propMts, paramAngs, sym, mtxf)
    end
end

promote_rule(::Type{Cnsolt{D,S,TA}}, ::Type{Cnsolt{D,S,TB}}) where {D,S,TA,TB} = Cnsolt{D,S,promote_type(TA,TB)}

struct ParallelFB{T,D} <: FilterBank{T,D}
    decimationFactor::NTuple{D, Int}
    nChannels::Int
    polyphaseOrder::NTuple{D, Int}

    analysisFilters::Vector{Array{T,D}}
    synthesisFilters::Vector{Array{T,D}}

    function ParallelFB(df::NTuple{D,Int}, nChs::Int, ppo::NTuple{D,Int}; dataType = Float64) where D
        szFilters = df .* (ppo .+ 1)
        afs = [ Array{dataType, D}(szFilters...) for p in 1:nChs ]
        sfs = [ Array{dataType, D}(szFilters...) for p in 1:nChs ]
        new{dataType, D}(df, nChs, ppo, afs, sfs)
    end
end

promote_rule(::Type{ParallelFB{TA,D}}, ::Type{ParallelFB{TB,D}}) where {TA,TB,D} = ParallelFB{promote_type(TA,TB),D}

struct MultiLayerCsc{T,D}
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

include("convexOptimization.jl")

include("blockproc.jl")
include("orthonormalMatrixSystem.jl")
include("polyphaseMatrices.jl")

include("analysisSystem.jl")
include("synthesisSystem.jl")

include("dictionaryLearning.jl")

include("view.jl")

include("mlcsc.jl")

end
