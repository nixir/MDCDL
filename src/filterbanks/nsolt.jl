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

    perm::NTuple{D, Int} # permutation of dimensions

    function RnsoltTypeI(::Type{T}, df::NTuple{D, Int}, ppo::NTuple{D, Int}, nchs::Tuple{Int, Int}; perm = (collect(1:D)...,), kwargs...) where {D,T}
        @assert isperm(perm) "invalid permutations"
        @assert (nchs[1] == nchs[2]) "channel size mismatch!"
        @assert (cld(prod(df),2) <= nchs[1] <= sum(nchs) - fld(prod(df),2)) && (fld(prod(df),2) <= nchs[2] <= sum(nchs) - cld(prod(df),2)) "invalid number of channels"
        # CJ = reverse(permdctmtx(T, df...), dims=2) |> Matrix
        CJ = permdctmtx(T, df...)

        W0 = Matrix{T}(I, nchs[1], nchs[1])
        U0 = Matrix{T}(I, nchs[2], nchs[2])
        Udks = ([
            [ (iseven(k) ? 1 : -1) * Matrix{T}(I, nchs[2], nchs[2]) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        new{T,D}(df, ppo, nchs, CJ, W0, U0, Udks, perm)
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

    perm::NTuple{D, Int}

    function RnsoltTypeII(::Type{T}, df::NTuple{D, Int}, ppo::NTuple{D, Int}, nchs::Tuple{Int, Int}; perm = (collect(1:D)...,), kwargs...) where {D,T}
        @assert isperm(perm) "invalid permutations"
        @assert (cld(prod(df),2) <= nchs[1] <= sum(nchs) - fld(prod(df),2)) && (fld(prod(df),2) <= nchs[2] <= sum(nchs) - cld(prod(df),2)) "invalid number of channels"
        # CJ = reverse(permdctmtx(T, df...), dims=2) |> Matrix
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

        new{T,D}(df, nStages, nchs, CJ, W0, U0, Wdks, Udks, perm)
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
    ??dks::NTuple{D, Vector{AbstractVector{T}}}

    ??::Diagonal

    perm::NTuple{D, Int}

    function CnsoltTypeI(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer; perm = (collect(1:D)...,), kwargs...) where {T,D}
        @assert isperm(perm) "invalid permutations"
        @assert (prod(df) <= sum(nchs)) "number of channels must be greater or equal to decimation factor"
        FJ = cdftmtx(T, df...)

        V0 = Matrix{T}(I, nchs, nchs)

        fP = fld(nchs, 2)
        Wdks = ([
            [ Matrix{T}(I, fP, fP) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        Udks = ([
            [ -Matrix{T}(I, fP, fP) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        ??dks = ([
            [ zeros(T, fld(nchs, 4)) for k in 1:ppo[d] ]
        for d in 1:D ]...,)

        ?? = Diagonal(cis.(zeros(nchs))) |> complex

        new{T,D}(df, ppo, nchs, FJ, V0, Wdks, Udks, ??dks, ??, perm)
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
    ??1dks::NTuple{D, Vector{AbstractVector{T}}}
    W??dks::NTuple{D, Vector{AbstractMatrix{T}}}
    U??dks::NTuple{D, Vector{AbstractMatrix{T}}}
    ??2dks::NTuple{D, Vector{AbstractVector{T}}}

    ??::Diagonal

    perm::NTuple{D, Int}

    function CnsoltTypeII(::Type{T}, df::NTuple{D,Int}, ppo::NTuple{D,Int}, nchs::Integer; perm = (collect(1:D)...,), kwargs...) where {T,D}
        @assert isperm(perm) "invalid permutations"
        @assert (prod(df) <= sum(nchs)) "number of channels must be greater or equal to decimation factor"
        @assert all(iseven.(ppo)) "polyphase order of each dimension must be odd"
        nStages = fld.(ppo, 2)
        FJ = cdftmtx(T, df...)

        V0 = Matrix{T}(I, nchs, nchs)

        fP, cP = fld(nchs,2), cld(nchs,2)
        Wdks = ([
            [ Matrix{T}(I, fP, fP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        Udks = ([
            [ -Matrix{T}(I, fP, fP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        ??1dks = ([
            [ zeros(T, fld(nchs, 4)) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        W??dks = ([
            [ Matrix{T}(I, cP, cP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        signU?? = Diagonal([fill(-1,cP-1)..., 1])
        U??dks = ([
            [ signU?? * Matrix{T}(I, cP, cP) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        ??2dks = ([
            [ zeros(T, fld(nchs, 4)) for k in 1:nStages[d] ]
        for d in 1:D ]...,)

        ?? = Diagonal(cis.(zeros(nchs))) |> complex

        new{T,D}(df, nStages, nchs, FJ, V0, Wdks, Udks, ??1dks, W??dks, U??dks, ??2dks, ??, perm)
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

function supertype_nsolt(::Type{NS}) where {NS<:AbstractNsolt}
    RNS = typeintersect(promote_type(NS, Rnsolt), AbstractNsolt)
    CNS = typeintersect(promote_type(NS, Cnsolt), AbstractNsolt)
    typeintersect(RNS, CNS)
end

function copy_params!(dst::NS, src::NS) where {NS<:RnsoltTypeI}
    mycp(a,b) = foreach(a, b) do ad, bd
        foreach((adk, bdk) -> adk .= bdk, ad, bd)
    end

    dst.CJ .= src.CJ

    dst.W0 .= src.W0
    dst.U0 .= src.U0

    mycp(dst.Udks, src.Udks)

    return dst
end

function copy_params!(dst::NS, src::NS) where {NS<:RnsoltTypeII}
    mycp(a,b) = foreach(a, b) do ad, bd
        foreach((adk, bdk) -> adk .= bdk, ad, bd)
    end

    dst.CJ .= src.CJ

    dst.W0 .= src.W0
    dst.U0 .= src.U0

    mycp(dst.Wdks, src.Wdks)
    mycp(dst.Udks, src.Udks)

    return dst
end

function copy_params!(dst::NS, src::NS) where {NS<:CnsoltTypeI}
    mycp(a,b) = foreach(a, b) do ad, bd
        foreach((adk, bdk) -> adk .= bdk, ad, bd)
    end

    dst.FJ .= src.FJ

    dst.V0 .= src.V0

    mycp(dst.Wdks, src.Wdks)
    mycp(dst.Udks, src.Udks)
    mycp(dst.??dks, src.??dks)

    dst.?? .= src.??

    return dst
end

function copy_params!(dst::NS, src::NS) where {NS<:CnsoltTypeII}
    mycp(a,b) = foreach(a, b) do ad, bd
        foreach((adk, bdk) -> adk .= bdk, ad, bd)
    end

    dst.FJ .= src.FJ

    dst.W0 .= src.W0
    dst.U0 .= src.U0

    mycp(dst.Wdks, src.Wdks)
    mycp(dst.Udks, src.Udks)
    mycp(dst.??1dks, src.??1dks)
    mycp(dst.W??dks, src.W??dks)
    mycp(dst.U??dks, src.U??dks)
    mycp(dst.??2dks, src.??2dks)

    dst.?? .= src.??

    return dst
end

Cnsolt(rn::RnsoltTypeI) = CnsoltTypeI(rn)

function CnsoltTypeI(rn::RnsoltTypeI{T}) where {T}
    Pw = rn.nChannels[1]
    Pu = rn.nChannels[2]
    M = prod(rn.decimationFactor)
    cM, fM = cld(M,2), fld(M,2)

    cn = CnsoltTypeI(T, decimations(rn), orders(rn), nchannels(rn))

    # cn.FJ .= diagm( 0 => [ ones(cM); 1im * ones(fM) ] ) * rn.CJ

    cn.V0 .= begin
        pms = [ collect(1:cM)...,
                  collect((1:Pw-cM) .+ M)...,
                  collect((1:fM) .+ cM)...,
                  collect((1:Pu-fM) .+ (Pw+fM))... ]

        pmtx = foldl(enumerate(pms), init=zero(cn.V0)) do mtx, (idx, pmi)
            setindex!(mtx, 1, idx, pmi)
        end

        # cat(rn.W0, rn.U0, dims=[1,2]) * pmtx
        sgnmtx = diagm( 0 => [ ones(cM); 1im * ones(fM) ] )
        C0 = cat(real( sgnmtx * rn.CJ * cn.FJ'), Matrix(I, (size(cn.V0) .- M)...), dims=[1,2])
        cat(rn.W0, rn.U0, dims=[1,2]) * pmtx * C0
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

    foreach(cn.??dks) do c??s
        foreach(c??s) do c??
            c?? .= zeros(T, size(c??)...)
        end
    end

    cn.?? .= Diagonal([ ones(Pw); -1im * ones(Pu) ])

    return cn
end

function show(io::IO, ::MIME"text/plain", nsolt::Rnsolt)
    print(io, "$(nsolt.nChannels)-channels $(typeof(nsolt)) with Decimation factor=$(nsolt.decimationFactor), Polyphase order=$(orders(nsolt))")
end

function show(io::IO, ::MIME"text/plain", nsolt::Cnsolt)
    print(io, "$(nsolt.nChannels)-channels $(typeof(nsolt)) with Decimation factor=$(nsolt.decimationFactor), Polyphase order=$(orders(nsolt))")
end

# permutedims_prop(n::AbstractNsolt, perm::AbstractVector) = permutedims_prop(n, (perm...,))
#
# function permutedims_prop(nsolt::NS, perm::NTuple{D,N}) where {T,D,NS<:AbstractNsolt{T,D},N<:Integer}
#     @assert isperm(perm) "invalid permutation"
#
#     Nsolt = supertype_nsolt(NS)
#     fp(v) = ([ v[perm[idx]] for idx = 1:D ]...,)
#     out = Nsolt(T, fp(decimations(nsolt)), fp(orders(nsolt)), nchannels(nsolt))
#
#     return setPermutatedParams!(out, nsolt, perm)
# end
#
# function setPermutatedParams!(dst::NS, src::NS, perm) where {NS<:RnsoltTypeI}
#     dst.CJ .= copy(src.CJ)
#     dst.U0 .= copy(src.U0)
#     dst.W0 .= copy(src.W0)
#     for (idxdst, idxsrc) in enumerate(perm)
#         dst.Udks[idxdst] .= copy(src.Udks[idxsrc])
#     end
#     dst
# end
#
# function setPermutatedParams!(dst::NS, src::NS, perm) where {NS<:RnsoltTypeII}
#     dst.CJ .= copy(src.CJ)
#     dst.U0 .= copy(src.U0)
#     dst.W0 .= copy(src.W0)
#     for (idxdst, idxsrc) in enumerate(perm)
#         dst.Udks[idxdst] .= copy(src.Udks[idxsrc])
#         dst.Wdks[idxdst] .= copy(src.Wdks[idxsrc])
#     end
#     dst
# end
#
# function setPermutatedParams!(dst::NS, src::NS, perm) where {NS<:CnsoltTypeI}
#     dst.FJ .= copy(src.FJ)
#     dst.V0 .= copy(src.V0)
#     for (idxdst, idxsrc) in enumerate(perm)
#         dst.Udks[idxdst] .= copy(src.Udks[idxsrc])
#         dst.Wdks[idxdst] .= copy(src.Wdks[idxsrc])
#         dst.??dks[idxdst] .= copy(src.??dks[idxsrc])
#     end
#     dst.?? .= copy(src.??)
#     dst
# end
#
# function setPermutatedParams!(dst::NS, src::NS, perm) where {NS<:CnsoltTypeII}
#     dst.FJ .= copy(src.FJ)
#     dst.V0 .= copy(src.V0)
#     for (idxdst, idxsrc) in enumerate(perm)
#         dst.Udks[idxdst] .= copy(src.Udks[idxsrc])
#         dst.Wdks[idxdst] .= copy(src.Wdks[idxsrc])
#         dst.??1dks[idxdst] .= copy(src.??1dks[idxsrc])
#         dst.U??dks[idxdst] .= copy(src.U??dks[idxsrc])
#         dst.W??dks[idxdst] .= copy(src.W??dks[idxsrc])
#         dst.??2dks[idxdst] .= copy(src.??2dks[idxsrc])
#     end
#     dst.?? .= copy(src.??)
#     dst
# end
