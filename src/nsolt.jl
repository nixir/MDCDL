abstract type AbstractNsolt{T,D} <: PolyphaseFB{T,D} end
abstract type Rnsolt{T,D} <: AbstractNsolt{T,D} end

Rnsolt(df::Integer, ppo::Integer, nChs; kwargs...) = Rnsolt(Float64, df, ppo, nChs; kwargs...)
Rnsolt(::Type{T}, df::Integer, ppo::Integer, nChs::Integer; kwargs...) where {T} = Rnsolt(T, df, ppo, (cld(nChs,2),fld(nChs,2)); kwargs...)
function Rnsolt(::Type{T}, df::Integer, ppo::Integer, nChs::Tuple{Int,Int}; dims::Integer=1, kwargs...) where {T}
    Rnsolt(T, (fill(df,dims)...,), (fill(ppo,dims)...,), nChs; kwargs...)
end

Rnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Union{Tuple{Int,Int}, Integer}; kwargs...) where {D} = Rnsolt(Float64, df, ppo, nChs; kwargs...)

Rnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Integer; kwargs...) where {D,T} = Rnsolt(T, df, ppo, (cld(nChs,2), fld(nChs,2)); kwargs...)

function Rnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Tuple{Int,Int}; kwargs...) where {T,D}
    return if nChs[1] == nChs[2]
        RnsoltTypeI(T, df, ppo, nChs; kwargs...)
    else
        RnsoltTypeII(T, df, ppo, nChs; kwargs...)
    end
end

struct RnsoltTypeI{T,D} <: Rnsolt{T,D}
    decimationFactor::NTuple{D, Int}
    nStages::NTuple{D, Int}
    nChannels::Tuple{Int, Int}

    CJ::AbstractMatrix # reverse of permutated DCT

    W0::AbstractMatrix{T} # parameter matrix of initial matrix
    U0::AbstractMatrix{T}

    Udks::NTuple{D, Vector{AbstractMatrix{T}}} # parameter matrices of propagation matrices

    function RnsoltTypeI(::Type{T}, df::NTuple{D, Int}, ppo::NTuple{D, Int}, nchs::Tuple{Int, Int}; kwargs...) where {D,T}
        @assert (nchs[1] == nchs[2]) "channel size mismatch!"
        CJ = permdctmtx(T, df...)

        W0 = Matrix{T}(I, nchs[1], nchs[1])
        U0 = Matrix{T}(I, nchs[2], nchs[2])
        Udks = ([
            [ (iseven(k) ? 1 : -1) * Matrix{T}(I, nchs[2], nchs[2]) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        new{T,D}(df, ppo, nchs, CJ, W0, U0, Udks)
    end
end

struct RnsoltTypeII{T,D} <: Rnsolt{T,D}
    decimationFactor::NTuple{D, Int}
    nStages::NTuple{D, Int}
    nChannels::Tuple{Int, Int}

    CJ::AbstractMatrix # reverse of permutated DCT

    # matricesW0U0Odd::AbstractArray{}

    W0::AbstractMatrix{T} # parameter matrix of initial matrix
    U0::AbstractMatrix{T} # parameter matrix of initial matrix

    Wdks::NTuple{D, Vector{AbstractMatrix{T}}} # parameter matrices of propagation matrices
    Udks::NTuple{D, Vector{AbstractMatrix{T}}} # parameter matrices of propagation matrices

    function RnsoltTypeII(::Type{T}, df::NTuple{D, Int}, ppo::NTuple{D, Int}, nchs::Tuple{Int, Int}; kwargs...) where {D,T}
        CJ = permdctmtx(T, df...)

        W0 = Matrix{T}(I, nchs[1], nchs[1])
        U0 = Matrix{T}(I, nchs[2], nchs[2])

        @assert all(iseven.(ppo)) "polyphase order of each dimension must be odd"
        nStages = fld.(ppo, 2)

        signW, signU = if nchs[1] < nchs[2]; (-1, 1) else (1, -1) end
        Wdks = ([
            [ signW * Matrix{T}(I, nchs[1], nchs[1]) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        Udks = ([
            [ signU * Matrix{T}(I, nchs[2], nchs[2]) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        new{T,D}(df, nStages, nchs, CJ, W0, U0, Wdks, Udks)
    end
end

abstract type Cnsolt{T,D} <: AbstractNsolt{T,D} end

Cnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Integer; kwargs...) where {D} = Cnsolt(Float64, df, ppo, nChs; kwargs...)
Cnsolt(df::Integer, ppo::Integer, nChs::Integer; kwargs...) = Cnsolt(Float64, df, ppo, nChs; kwargs...)

function Cnsolt(::Type{T}, df::Integer, ppo::Integer, nChs::Integer; dims=1, kwargs...) where {T}
    Cnsolt(T, (fill(df, dims)...,), (fill(ppo, dims)...,), nChs; kwargs...)
end

function Cnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Integer; kwargs...) where {T,D}
    return if iseven(nChs)
        CnsoltTypeI(T, df, ppo, nChs; kwargs...)
    else
        CnsoltTypeII(T, df, ppo, nChs; kwargs...)
    end
end

struct CnsoltTypeI{T,D} <: Cnsolt{T,D}
    decimationFactor::NTuple{D,Int}
    nStages::NTuple{D,Int}
    nChannels::Integer

    FJ::AbstractMatrix

    V0::AbstractMatrix{T}

    Wdks::NTuple{D, Vector{AbstractMatrix{T}}}
    Udks::NTuple{D, Vector{AbstractMatrix{T}}}
    θdks::NTuple{D, Vector{AbstractVector{T}}}

    Φ::Diagonal

    function CnsoltTypeI(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Integer) where {T,D}
        FJ = cdftmtx(T, df...)

        V0 = Matrix{T}(I, nChs, nChs)

        fP = fld(nChs, 2)
        Wdks = ([
            [ Matrix{T}(I, fP, fP) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        Udks = ([
            [ -Matrix{T}(I, fP, fP) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        θdks = ([
            [ zeros(T, fld(nChs, 4)) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        Φ = Diagonal{T}(cis.(zeros(nChs)))

        new{T,D}(df, ppo, nChs, FJ, V0, Wdks, Udks, θdks, Φ)
    end
end

struct CnsoltTypeII{T,D} <: Cnsolt{T,D}
    decimationFactor::NTuple{D,Int}
    nStages::NTuple{D,Int}
    nChannels::Integer

    FJ::AbstractMatrix

    V0::AbstractMatrix{T}

    Wdks::NTuple{D, Vector{AbstractMatrix{T}}}
    Udks::NTuple{D, Vector{AbstractMatrix{T}}}
    θ1dks::NTuple{D, Vector{AbstractVector{T}}}
    Ŵdks::NTuple{D, Vector{AbstractMatrix{T}}}
    Ûdks::NTuple{D, Vector{AbstractMatrix{T}}}
    θ2dks::NTuple{D, Vector{AbstractVector{T}}}

    Φ::Diagonal

    function CnsoltTypeII(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nChs::Integer) where {T,D}
        nStages = fld.(ppo, 2)
        FJ = cdftmtx(T, df...)

        V0 = Matrix{T}(I, nChs, nChs)

        fP, cP = fld(nChs,2), cld(nChs,2)
        Wdks = ([
            [ Matrix{T}(I, fP, fP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        Udks = ([
            [ -Matrix{T}(I, fP, fP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        θ1dks = ([
            [ zeros(T, fld(nChs, 4)) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        Ŵdks = ([
            [ Matrix{T}(I, cP, cP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        signÛ = Diagonal([fill(-1,cP-1)..., 1])
        Ûdks = ([
            [ signÛ * Matrix{T}(I, cP, cP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        θ2dks = ([
            [ zeros(T, fld(nChs, 4)) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        Φ = Diagonal(cis.(zeros(nChs)))

        new{T,D}(df, nStages, nChs, FJ, V0, Wdks, Udks, θ1dks, Ŵdks, Ûdks, θ2dks, Φ)
    end
end

# promote_rule(::Type{Cnsolt{TA,D}}, ::Type{Cnsolt{TB,D}}) where {D,TA,TB} = Cnsolt{promote_type(TA,TB),D}

similar(nsolt::Cnsolt{T,DS}, element_type::Type=T, df::NTuple{DD}=decimations(nsolt), ord::NTuple{DD}=orders(nsolt), nch::Integer=nchannels(nsolt)) where {T,DS,DD} = Cnsolt(element_type, df, ord, nch)

orders(fb::RnsoltTypeI) = fb.nStages
orders(fb::CnsoltTypeI) = fb.nStages
orders(fb::RnsoltTypeII) = 2 .* fb.nStages
orders(fb::CnsoltTypeII) = 2 .* fb.nStages

istype1(::Type{RnsoltTypeI}) = true
istype1(::Type{CnsoltTypeI}) = true
istype1(::Type{RnsoltTypeII}) = false
istype1(::Type{CnsoltTypeII}) = false

istype1(::T) where {T<:AbstractNsolt} = istype1(T)

istype2(::Type{T}) where {T<:AbstractNsolt} = !istype1(T)
istype2(::T) where {T<:AbstractNsolt} = !istype1(T)
