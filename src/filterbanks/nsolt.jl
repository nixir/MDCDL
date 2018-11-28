import Base: summary, show

abstract type AbstractNsolt{T,D} <: PolyphaseFB{T,D} end
abstract type Rnsolt{T,D} <: AbstractNsolt{T,D} end

Rnsolt(df::Integer, ppo::Integer, nchs; kwargs...) = Rnsolt(Float64, df, ppo, nchs; kwargs...)
Rnsolt(::Type{T}, df::Integer, ppo::Integer, nchs::Integer; kwargs...) where {T} = Rnsolt(T, df, ppo, (cld(nchs,2),fld(nchs,2)); kwargs...)
function Rnsolt(::Type{T}, df::Integer, ppo::Integer, nchs::Tuple{Int,Int}; dims::Integer=1, kwargs...) where {T}
    Rnsolt(T, (fill(df,dims)...,), (fill(ppo,dims)...,), nchs; kwargs...)
end

Rnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Union{Tuple{Int,Int}, Integer}; kwargs...) where {D} = Rnsolt(Float64, df, ppo, nchs; kwargs...)

Rnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer; kwargs...) where {D,T} = Rnsolt(T, df, ppo, (cld(nchs,2), fld(nchs,2)); kwargs...)

function Rnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Tuple{Int,Int}; kwargs...) where {T,D}
    return if nchs[1] == nchs[2]
        RnsoltTypeI(T, df, ppo, nchs; kwargs...)
    else
        RnsoltTypeII(T, df, ppo, nchs; kwargs...)
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
        @assert (cld(prod(df),2) <= nchs[1] <= sum(nchs) - fld(prod(df),2)) && (fld(prod(df),2) <= nchs[2] <= sum(nchs) - cld(prod(df),2)) "invalid number of channels"
        CJ = reverse(permdctmtx(T, df...), dims=2) |> Matrix

        W0 = Matrix{T}(I, nchs[1], nchs[1])
        U0 = Matrix{T}(I, nchs[2], nchs[2])
        Udks = ([
            [ (iseven(k) ? 1 : -1) * Matrix{T}(I, nchs[2], nchs[2]) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        new{T,D}(df, ppo, nchs, CJ, W0, U0, Udks)
    end
end

RnsoltTypeI(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs; kwargs...) where {D} = RnsoltTypeI(Float64, df, ppo, nchs; kwargs...)
RnsoltTypeI(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer; kwargs...) where {T,D} = RnsoltTypeI(T, df, ppo, (cld(nchs,2), fld(nchs,2)); kwargs...)

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
        @assert (cld(prod(df),2) <= nchs[1] <= sum(nchs) - fld(prod(df),2)) && (fld(prod(df),2) <= nchs[2] <= sum(nchs) - cld(prod(df),2)) "invalid number of channels"
        CJ = reverse(permdctmtx(T, df...), dims=2) |> Matrix

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

RnsoltTypeII(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs; kwargs...) where {D} = RnsoltTypeII(Float64, df, ppo, nchs; kwargs...)
RnsoltTypeII(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer; kwargs...) where {T,D} = RnsoltTypeII(T, df, ppo, (cld(nchs,2), fld(nchs,2)); kwargs...)

abstract type Cnsolt{T,D} <: AbstractNsolt{T,D} end

Cnsolt(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer; kwargs...) where {D} = Cnsolt(Float64, df, ppo, nchs; kwargs...)
Cnsolt(df::Integer, ppo::Integer, nchs::Integer; kwargs...) = Cnsolt(Float64, df, ppo, nchs; kwargs...)

function Cnsolt(::Type{T}, df::Integer, ppo::Integer, nchs::Integer; dims=1, kwargs...) where {T}
    Cnsolt(T, (fill(df, dims)...,), (fill(ppo, dims)...,), nchs; kwargs...)
end

function Cnsolt(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer; kwargs...) where {T,D}
    return if iseven(nchs)
        CnsoltTypeI(T, df, ppo, nchs; kwargs...)
    else
        CnsoltTypeII(T, df, ppo, nchs; kwargs...)
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

    function CnsoltTypeI(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer) where {T,D}
        @assert (prod(df) <= sum(nchs)) "number of channels must be greater or equal to decimation factor"
        FJ = reverse(cdftmtx(T, df...), dims=2) |> Matrix

        V0 = Matrix{T}(I, nchs, nchs)

        fP = fld(nchs, 2)
        Wdks = ([
            [ Matrix{T}(I, fP, fP) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        Udks = ([
            [ -Matrix{T}(I, fP, fP) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        θdks = ([
            [ zeros(T, fld(nchs, 4)) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        Φ = Diagonal(cis.(zeros(nchs))) |> complex

        new{T,D}(df, ppo, nchs, FJ, V0, Wdks, Udks, θdks, Φ)
    end
end

CnsoltTypeI(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer; kwargs...) where {D} = CnsoltTypeI(Float64, df, ppo, nchs; kwargs...)

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

    function CnsoltTypeII(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer) where {T,D}
        @assert (prod(df) <= sum(nchs)) "number of channels must be greater or equal to decimation factor"
        @assert all(iseven.(ppo)) "polyphase order of each dimension must be odd"
        nStages = fld.(ppo, 2)
        FJ = reverse(cdftmtx(T, df...), dims=2) |> Matrix

        V0 = Matrix{T}(I, nchs, nchs)

        fP, cP = fld(nchs,2), cld(nchs,2)
        Wdks = ([
            [ Matrix{T}(I, fP, fP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        Udks = ([
            [ -Matrix{T}(I, fP, fP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        θ1dks = ([
            [ zeros(T, fld(nchs, 4)) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        Ŵdks = ([
            [ Matrix{T}(I, cP, cP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        signÛ = Diagonal([fill(-1,cP-1)..., 1])
        Ûdks = ([
            [ signÛ * Matrix{T}(I, cP, cP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        θ2dks = ([
            [ zeros(T, fld(nchs, 4)) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        Φ = Diagonal(cis.(zeros(nchs))) |> complex

        new{T,D}(df, nStages, nchs, FJ, V0, Wdks, Udks, θ1dks, Ŵdks, Ûdks, θ2dks, Φ)
    end
end

CnsoltTypeII(df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer; kwargs...) where {D} = CnsoltTypeII(Float64, df, ppo, nchs; kwargs...)

promote_rule(::Type{RnsoltTypeI{TA,D}}, ::Type{RnsoltTypeI{TB,D}}) where {D,TA,TB} = RnsoltTypeI{promote_rule(TA,TB),D}

promote_rule(::Type{RnsoltTypeII{TA,D}}, ::Type{RnsoltTypeII{TB,D}}) where {D,TA,TB} = RnsoltTypeII{promote_rule(TA,TB),D}

promote_rule(::Type{CnsoltTypeI{TA,D}}, ::Type{CnsoltTypeI{TB,D}}) where {D,TA,TB} = CnsoltTypeI{promote_rule(TA,TB),D}

promote_rule(::Type{CnsoltTypeII{TA,D}}, ::Type{CnsoltTypeII{TB,D}}) where {D,TA,TB} = CnsoltTypeII{promote_type(TA,TB),D}

similar(nsolt::Rnsolt{T,DS}, element_type::Type=T, df::NTuple{DD}=decimations(nsolt), ord::NTuple{DD}=orders(nsolt), nch=nchannels_sa(nsolt)) where {T,DS,DD} = Rnsolt(element_type, df, ord, nch)

similar(nsolt::Cnsolt{T,DS}, element_type::Type=T, df::NTuple{DD}=decimations(nsolt), ord::NTuple{DD}=orders(nsolt), nch=nchannels(nsolt)) where {T,DS,DD} = Cnsolt(element_type, df, ord, nch)

nchannels_sa(nsolt::AbstractNsolt) = nchannels(nsolt)
nchannels_sa(nsolt::Rnsolt) = nsolt.nChannels

orders(fb::RnsoltTypeI) = fb.nStages
orders(fb::CnsoltTypeI) = fb.nStages
orders(fb::RnsoltTypeII) = 2 .* fb.nStages
orders(fb::CnsoltTypeII) = 2 .* fb.nStages

nstages(fb::AbstractNsolt) = fb.nStages

istype1(::Type{NS}) where {NS<:RnsoltTypeI} = true
istype1(::Type{NS}) where {NS<:CnsoltTypeI} = true
istype1(::Type{NS}) where {NS<:RnsoltTypeII} = false
istype1(::Type{NS}) where {NS<:CnsoltTypeII} = false

istype1(::T) where {T<:AbstractNsolt} = istype1(T)

istype2(::Type{T}) where {T<:AbstractNsolt} = !istype1(T)
istype2(::T) where {T<:AbstractNsolt} = !istype1(T)

Cnsolt(rn::RnsoltTypeI) = CnsoltTypeI(rn)

function CnsoltTypeI(rn::RnsoltTypeI{T}) where {T}
    Pw = rn.nChannels[1]
    Pu = rn.nChannels[2]
    M = prod(rn.decimationFactor)
    cM, fM = cld(M,2), fld(M,2)

    cn = CnsoltTypeI(T, decimations(rn), orders(rn), nchannels(rn))

    cn.FJ .= diagm( 0 => [ ones(cM); 1im * ones(fM) ] ) * rn.CJ

    cn.V0 .= begin
        perms = [ collect(1:cM)...,
                  collect((1:Pw-cM) .+ M)...,
                  collect((1:fM) .+ cM)...,
                  collect((1:Pu-fM) .+ (Pw+fM))... ]

        pmtx = foldl(enumerate(perms), init=zero(cn.V0)) do mtx, (idx, pmi)
            setindex!(mtx, 1, idx, pmi)
        end

        cat(rn.W0, rn.U0, dims=[1,2]) * pmtx
    end

    foreach(cn.Wdks) do cWs
        foreach(cWs) do cW
            cW .= Matrix{T}(I, size(cW)...)
        end
    end

    foreach(cn.Udks, rn.Udks) do cUs, rUs
        foreach(cUs, rUs) do cU, rU
            cU .= rU
        end
    end

    foreach(cn.θdks) do cθs
        foreach(cθs) do cθ
            cθ .= zeros(T, size(cθ)...)
        end
    end

    cn.Φ .= Diagonal([ ones(Pw); -1im * ones(Pu) ])

    return cn
end

function show(io::IO, ::MIME"text/plain", nsolt::Rnsolt)
    print(io, "$(nsolt.nChannels)-channels $(typeof(nsolt)) with Decimation factor=$(nsolt.decimationFactor), Polyphase order=$(orders(nsolt))")
end

function show(io::IO, ::MIME"text/plain", nsolt::Cnsolt)
    print(io, "$(nsolt.nChannels)-channels $(typeof(nsolt)) with Decimation factor=$(nsolt.decimationFactor), Polyphase order=$(orders(nsolt))")
end
