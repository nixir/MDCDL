module MDCDL # Multi-Dimensional Convolutional Dictionary Learning

# A type of D-dimensional CNSOLT

export PolyphaseVector
export FilterBank, PolyphaseFB, ParallelFB, Cnsolt, Rnsolt
export MultiLayerCsc
export analyze, synthesize
# export cconv
# export upsample, downsample
export atmimshow
# export getAnalysisBank
export getAnalysisFilters, getSynthesisFilters
export mdfilter
export getAngleParameters, setAngleParameters!

include("basicComplexDSP.jl")
import Base.promote_rule

struct PolyphaseVector{T,D} <: AbstractArray{T,2}
    data::Matrix{T}
    nBlocks::NTuple{D, Int64}
end

promote_rule(::Type{PolyphaseVector{TA,D}}, ::Type{PolyphaseVector{TB,D}}) where {TA,TB,D} = PolyphaseVector{promote_type(TA,TB), D}

abstract type FilterBank{T,D} end
abstract type PolyphaseFB{T,D} <: FilterBank{T,D} end

struct Rnsolt{D,S,T} <: PolyphaseFB{T,D}
    decimationFactor::NTuple{D, Int64}
    nChannels::Tuple{Int64,Int64}
    polyphaseOrder::NTuple{D, Int64}

    nVanishingMoment::Int64

    initMatrices::Array{Matrix{T},1}
    propMatrices::Array{Array{Matrix{T},1},1}

    matrixC::Matrix{T}

    function Rnsolt(df::NTuple{D,Int64}, nChs::Tuple{Int64, Int64}, ppo::NTuple{D,Int64}; vanishingMoment::Int64 = 0, dataType = Float64) where D
        P = sum(nChs)
        structType = (nChs[1] == nChs[2]) ? 1 : 2
        if prod(df) > P
            error("The number of channels must be equal or greater than a product of the decimation factor.")
        end
        if vanishingMoment != 0 && vanishingMoment != 1
            error("The number of vanishing moments must be 0 or 1.")
        end
        if structType == 2
            error("Sorry, the Type-II RNSOLT hasn't implemented yet. Use Type-I.")
        end
        # dirPerm = ntuple(d -> d, D)
        initMts = Array[ eye(dataType, p) for p in nChs ]
        propMts = Array[ Array[ (n % 2 == 0 ? 1 : -1) * eye(dataType,nChs[1]) for n in 1:ppo[pd] ] for pd in 1:D ] # Consider only Type-I NSOLT

        mtxc = flipdim(MDCDL.permdctmtx(df...),2)

        new{D,structType,dataType}(df, nChs, ppo, vanishingMoment, initMts, propMts, mtxc)
    end
end

promote_rule(::Type{Rnsolt{D,S,TA}}, ::Type{Rnsolt{D,S,TB}}) where {D,S,TA,TB} = Rnsolt{D,S,promote_type(TA,TB)}

struct Cnsolt{D,S,T} <: PolyphaseFB{Complex{T},D}
    decimationFactor::NTuple{D, Int64}
    nChannels::Int64
    polyphaseOrder::NTuple{D, Int64}
    # directionPermutation::NTuple{D, Int64}

    nVanishingMoment::Int64 # reserved for furture works

    initMatrices::Array{Matrix{T},1}
    propMatrices::Array{Array{Matrix{T},1},1}
    paramAngles::Array{Array{Vector{T},1},1}
    symmetry::Diagonal{Complex{T}}
    matrixF::Matrix{Complex{T}}

    # constructor
    function Cnsolt(df::NTuple{D,Int64}, nChs::Int64, ppo::NTuple{D,Int64}; vanishingMoment::Int64 = 0, dataType = Float64) where D
        structType = (nChs % 2 == 0) ? 1 : 2
        if prod(df) > nChs
            error("The number of channels must be equal or greater than a product of the decimation factor.")
        end
        if vanishingMoment != 0 && vanishingMoment != 1
            error("The number of vanishing moments must be 0 or 1.")
        end
        initMts = Array[ eye(dataType,nChs) ]
        propMts = Array[ Array[ (n % 2 == 0 ? -1 : 1) * eye(dataType,fld(nChs,2)) for n in 1:2*ppo[pd] ] for pd in 1:D ] # Consider only Type-I CNSOLT
        paramAngs = Array[ Array[ zeros(dataType,fld(nChs,4)) for n in 1:ppo[pd] ] for pd in 1:D ]
        sym = Diagonal{Complex{dataType}}(ones(nChs))
        mtxf = flipdim(MDCDL.cdftmtx(df...),2)

        new{D,structType,dataType}(df, nChs, ppo, vanishingMoment, initMts, propMts, paramAngs, sym, mtxf)
    end
end

promote_rule(::Type{Cnsolt{D,S,TA}}, ::Type{Cnsolt{D,S,TB}}) where {D,S,TA,TB} = Cnsolt{D,S,promote_type(TA,TB)}

struct ParallelFB{T,D} <: FilterBank{T,D}
    decimationFactor::NTuple{D, Int64}
    nChannels::Int64
    polyphaseOrder::NTuple{D, Int64}

    analysisFilters::Vector{Array{T,D}}
    synthesisFilters::Vector{Array{T,D}}

    function ParallelFB(df::NTuple{D,Int64}, nChs::Int64, ppo::NTuple{D,Int64}; dataType = Float64) where D
        szFilters = df .* (ppo .+ 1)
        afs = [ Array{dataType, D}(szFilters...) for p in 1:nChs ]
        sfs = [ Array{dataType, D}(szFilters...) for p in 1:nChs ]
        new{dataType, D}(df, nChs, ppo, afs, sfs)
    end
end

promote_rule(::Type{ParallelFB{TA,D}}, ::Type{ParallelFB{TB,D}}) where {TA,TB,D} = ParallelFB{promote_type(TA,TB),D}

struct MultiLayerCsc{T,D}
    nLayers::Int64

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
