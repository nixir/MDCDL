using TiledIteration: TileIterator
using FFTW: fft, dct

function upsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D} = (fill(0,D)...,)) where {T,D}
    @assert all(0 .<= offset .< factor) "offset is out of range"
    szout = size(x) .* factor
    setindex!(zeros(T, szout), x, StepRange.(offset .+ 1, factor, szout)...)
end

function downsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D}=(fill(0,D)...,)) where {T,D}
    @assert all(0 .<= offset .< factor) "offset is out of range"
    x[StepRange.(offset .+ 1, factor, size(x))...]
end

representationmatrix(f, sz::NTuple) = representationmatrix(f, sz...)
function representationmatrix(f, sz::Integer...)
    hcat([ setindex!(zeros(sz), 1, idx) |> f |> vec for idx in 1:prod(sz) ]...)
end

Base.@pure function haarbasis(d::Integer)
    w = walsh(d)
    @views [ reshape(w[p,:], fill(2,d)...) |> Array for p in 1:size(w, 1)]
end

Base.@pure function walsh(n::Integer)
    # ifelse(n >= 0, sub_walsh(Val(n)), error("n must to be a positive")) # this code is not work correctly.
    if n >= 0; sub_walsh(Val(n)) else error("n must to be a positive") end
end

function sub_walsh(::Val{N}) where {N}
    w = sub_walsh(Val(N-1))
    return [ w w ; w -w ]
end
sub_walsh(::Val{0}) = 1

# matrix-formed CDFT operator for D-dimensional signal
cdftmtx(sz::NTuple) = cdftmtx(sz...)
cdftmtx(sz::Integer...) = cdftmtx(Float64, sz...)
cdftmtx(T::Type, sz::NTuple) = cdftmtx(T, sz...)
cdftmtx(::Type, sz::Integer...) = cdftmtx(Float64, sz...)

cdftmtx(::Type{Complex{T}}, sz...) where {T} = cdftmtx(T, sz...)

Base.@pure function cdftmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    len = prod(sz)

    mtx = representationmatrix(x->fft(T.(x)), sz)
    rm = Diagonal(Complex{T}[ exp(-1im*angle(mtx[n,end])/2) for n in 1:len ])

    complex(T).(rm * mtx / sqrt(len))
end

permdctmtx(sz::NTuple) = permdctmtx(sz...)
permdctmtx(sz::Integer...) = permdctmtx(Float64, sz...)
permdctmtx(T::Type, sz::NTuple) = permdctmtx(T, sz...)

permdctmtx(::Type, sz::Integer...) = permdctmtx(Float64, sz...)

Base.@pure function permdctmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    mtx = representationmatrix(x->dct(T.(x)), sz)

    isevenids = map(ci->iseven(sum(ci.I .- 1)), CartesianIndices(sz)) |> vec
    permids = sortperm(isevenids; rev=true, alg=Base.DEFAULT_STABLE)

    @views vcat([ transpose(mtx[pi,:]) for pi in permids ]...)
end

function getMatrixB(P::Integer, angs::AbstractVector{T}) where T
    @assert (length(angs) == fld(P,4)) "mismatch number of channels"
    hP = fld(P,2)
    psangs = (2 .* angs .+ pi) ./ 4
    ss, cs = sin.(psangs), cos.(psangs)

    LC = map(ss, cs) do s, c
        [ (-1im*c) (-1im*s); c (-s) ]
    end
    LS = map(ss, cs) do s, c
        [ s c; (1im*s) (-1im*c) ]
    end

    pbm = ones(fill(hP % 2,2)...)
    C = cat(LC..., pbm; dims=[1,2])
    S = cat(LS..., 1im*pbm; dims=[1,2])

    [ C conj(C); S conj(S) ] / sqrt(convert(T,2))
end

function analysisbank(nsolt::AbstractNsolt)
    M = prod(decimations(nsolt))
    ord = orders(nsolt)

    # create inpulse signal matrix
    mtx0 = reverse(Matrix(I, M, M .* prod(ord .+ 1) ), dims=1)
    krncenter = initialStep(nsolt, mtx0 )

    nStrides = (cumprod([ M, (ord[1:end-1] .+ 1)... ])...,)
    rotdimsfcns = (fill(identity, ndims(nsolt))...,)
    krnsym = extendAtoms(nsolt, krncenter, nStrides, rotdimsfcns, border=:circular_traditional)

    return shiftFilterSymmetry(nsolt, krnsym)
end

# compatible mode for SaivDr
# function analysisbank_compatible(nsolt::AbstractNsolt)
# #function analysisbank(nsolt::AbstractNsolt)
#     M = prod(decimations(nsolt))
#     ord = orders(nsolt)
#
#     # create inpulse signal matrix
#     mtx0 = reverse(Matrix(I, M, M .* prod(ord .+ 1) ), dims=1)
#     # mtx0 = circshift(mtx0, (0, -M))
#     krncenter = initialStep(nsolt, mtx0 )
#
#     nStrides = (cumprod([ M, (ord[1:end-1] .+ 1)... ])...,)
#     rotdimsfcns = (fill(identity, ndims(nsolt))...,)
#     krnsym = extendAtoms(nsolt, krncenter, nStrides, rotdimsfcns, border=:circular_traditional)
#
#     return shiftFilterSymmetry(nsolt, krnsym)
# end

kernels(pfb::PolyphaseFB) = (analysiskernels(pfb), synthesiskernels(pfb))

function analysiskernels(pfb::PolyphaseFB)
    df = decimations(pfb)
    afb = analysisbank(pfb)

    @views map([ reshape(afb[p,:], prod(df), :) for p in 1:size(afb, 1)]) do vf
        out = similar(vf, kernelsize(pfb)...)
        for (idx, tile) in enumerate(TileIterator(axes(out), df))
            out[tile...] = reshape(vf[:, idx], df...)
        end
        out
    end
end

function synthesiskernels(cc::AbstractNsolt)
    map(analysiskernels(cc)) do af
        reshape(af .|> conj |> vec |> reverse, size(af))
    end
end

function mdarray2polyphase(x::AbstractArray{TX,D}, szBlock::NTuple{D,TS}) where {TX,D,TS<:Integer}
    nBlocks = fld.(size(x), szBlock)

    @assert all(size(x) .% szBlock .== 0) "size error. input data: $(size(x)), block size: $(szBlock)."

    # outdata = hcat([ vec(@view x[tile...]) for tile in TileIterator(axes(x), szBlock)]...)
    outdata = similar(x, prod(szBlock), prod(nBlocks))
    @views for (idx, tile) in enumerate(TileIterator(axes(x), szBlock))
        outdata[:,idx] = vec(x[tile...])
    end
    PolyphaseVector(outdata, nBlocks)
end

function polyphase2mdarray(x::PolyphaseVector{TX,D}, szBlock::NTuple{D,TS}) where {TX,D,TS<:Integer}
    @assert (size(x.data, 1) == prod(szBlock)) "size mismatch! 'prod(szBlock)' must be equal to $(size(x.data,1))."

    out = similar(x.data, (x.nBlocks .* szBlock)...)
    @views for (idx, tile) in enumerate(TileIterator(axes(out), szBlock))
        out[tile...] = reshape(x.data[:,idx], szBlock...)
    end
    out
end

function shiftdimspv(x::AbstractMatrix, nBlocks::Integer)
    @views hcat([ x[:, (1:nBlocks:end) .+ idx] for idx = 0:nBlocks-1 ]...)
end

ishiftdimspv(x::AbstractMatrix, nBlocks::Integer) = shiftdimspv(x, fld(size(x, 2), nBlocks))

@inline function unnormalized_butterfly!(xu::T, xl::T) where {T<:AbstractMatrix}
    tu, tl = (xu + xl, xu - xl)
    xu .= tu
    xl .= tl
    nothing
end

@inline function half_butterfly!(xu::T, xl::T) where {T<:AbstractMatrix}
    tu, tl = (xu + xl, xu - xl) ./ 2
    xu .= tu
    xl .= tl
    nothing
end

function shiftcoefs!(V::Val{:circular}, k::Integer, mtxup::AbstractMatrix, mtxlw::AbstractMatrix, nShift::Integer)
    if isodd(k)
        shiftcoefs_odd!(V, mtxlw, nShift)
    else
        shiftcoefs_even!(V, mtxup, nShift)
    end
    nothing
end

function shiftcoefs_odd!(::Val{:circular}, mtx::AbstractMatrix, nShift::Integer)
    mtx .= circshift(mtx, (0, nShift))
end

shiftcoefs_even!(V::Val{:circular}, mtx, nShift) = shiftcoefs_odd!(V, mtx, -nShift)

adjshiftcoefs!(v::Val{:circular}, k, mtxup, mtxlw, nShift::Integer) = shiftcoefs!(v, k, mtxup, mtxlw, -nShift)

# function shiftcoefs!(::Val{:zero}, k::Integer, mtxup::AbstractMatrix, mtxlw::AbstractMatrix, nShift::Integer)
#     if isodd(k)
#         mtxlw[:, 1+nShift:end] .= @view mtxlw[:, 1:end-nShift]
#         mtxlw[:, 1:nShift] .= 0
#     else
#         mtxup[:, 1:end-nShift] .= @view mtxup[:, 1+nShift:end]
#         mtxup[:, end-nShift+1:end] .= 0
#     end
#     nothing
# end
#
# function adjshiftcoefs!(::Val{:zero}, k::Integer, mtxup::AbstractMatrix, mtxlw::AbstractMatrix, nShift::Integer)
#     if isodd(k)
#         mtxlw[:, 1:end-nShift] .= @view mtxlw[:, 1+nShift:end]
#         mtxlw[:, end-nShift+1:end] .= 0
#     else
#         mtxup[:, 1+nShift:end] .= @view mtxup[:, 1:end-nShift]
#         mtxup[:, 1:nShift] .= 0
#     end
#     nothing
# end

function shiftcoefs!(::Val{:circular_traditional}, ::Integer, ::Any, mtxlw::AbstractMatrix, nShift::Integer)
    mtxlw .= circshift(mtxlw, (0, nShift))
end

adjshiftcoefs!(v::Val{:circular_traditional}, k, mtxup, mtxlw, nShift::Integer) = shiftcoefs!(v, k, mtxup, mtxlw, -nShift)
