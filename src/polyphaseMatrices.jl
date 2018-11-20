using TiledIteration: TileIterator
using FFTW: fft, dct

function upsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D} = (fill(0,D)...,)) where {T,D}
    szout = size(x) .* factor
    setindex!(zeros(T, szout), x, StepRange.(offset .+ 1, factor, szout)...)
end

function downsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D}=(fill(0,D)...,)) where {T,D}
    x[StepRange.(offset .+ 1, factor, size(x))...]
end

representationmatrix(f, sz::NTuple) = representationmatrix(f, sz...)
function representationmatrix(f::Function, sz::Integer...)
    hcat([ setindex!(zeros(sz), 1, idx) |> f |> vec for idx in 1:prod(sz) ]...)
end

Base.@pure function haarbasis(d::Integer)
    w = walsh(d)
    [ reshape(@view(w[p,:]), fill(2,d)...) |> Array for p in 1:size(w, 1)]
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

    vcat([ transpose(@view mtx[pi,:]) for pi in permids ]...)
end

function getMatrixB(P::Integer, angs::AbstractVector{T}) where T
    hP = fld(P,2)
    psangs = (2 .* angs .+ pi) ./ 4
    cs = cos.(psangs)
    ss = sin.(psangs)

    LC = [[ (-1im*cs[n]) (-1im*ss[n]); (cs[n]) (-ss[n]) ] for n in 1:fld(hP,2) ]
    LS = [[ (ss[n]) (cs[n]); (1im*ss[n]) (-1im*cs[n]) ] for n in 1:fld(hP,2) ]

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

kernels(pfb::PolyphaseFB) = (analysiskernels(pfb), synthesiskernels(pfb))

function analysiskernels(pfb::PolyphaseFB)
    df = decimations(pfb)
    afb = analysisbank(pfb)

    return map([ @view(afb[p,:]) for p in 1:size(afb, 1) ]) do vf
        rspvf = reshape(vf, prod(df), :)
        out = similar(vf, kernelsize(pfb)...)
        tilesout = TileIterator(axes(out), df)

        foldl(enumerate(tilesout), init=out) do mtx, (idx, tile)
            setindex!(mtx, reshape(@view(rspvf[:, idx]), df...), tile...)
        end
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

    # outdata = hcat([ vec(@view x[tile...]) for tile in TileIterator(axes(x), szBlocks)]...)
    outdata = similar(x, prod(szBlock), prod(nBlocks))
    tiles = collect(TileIterator(axes(x), szBlock))
    for idx in 1:length(tiles)
        outdata[:,idx] = vec(@view x[tiles[idx]...])
    end
    PolyphaseVector(outdata, nBlocks)
end

function mdarray2polyphase(x::AbstractArray{T,D}) where {T,D}
    nBlocks = size(x)[1:D-1]
    outdata = transpose(reshape(x, prod(nBlocks), size(x,D)))
    PolyphaseVector(outdata, nBlocks)
end

function polyphase2mdarray(x::PolyphaseVector{TX,D}, szBlock::NTuple{D,TS}) where {TX,D,TS<:Integer}
    @assert (size(x.data, 1) == prod(szBlock)) "size mismatch! 'prod(szBlock)' must be equal to $(size(x.data,1))."

    out = similar(x.data, (x.nBlocks .* szBlock)...)
    for (idx, tile) in enumerate(TileIterator(axes(out), szBlock))
        out[tile...] = reshape(@view(x.data[:,idx]), szBlock...)
    end
    out
end

function polyphase2mdarray(x::PolyphaseVector)
    reshape(transpose(x.data), x.nBlocks..., size(x.data, 1))
end

function shiftdimspv(x::AbstractMatrix, nBlocks::Integer)
    hcat([ @view x[:, (1:nBlocks:end) .+ idx] for idx = 0:nBlocks-1 ]...)
end

ishiftdimspv(x::AbstractMatrix, nBlocks::Integer) = shiftdimspv(x, fld(size(x, 2), nBlocks))

function butterfly!(x::AbstractMatrix, p::Integer)
    xu = x[1:p,:]
    xl = x[end-(p-1):end,:]

    x[1:p,:]           .= (xu + xl) / sqrt(2)
    x[end-(p-1):end,:] .= (xu - xl) / sqrt(2)
end

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

function shiftcoefs!(::Val{:circular}, k::Integer, mtxup::AbstractMatrix, mtxlw::AbstractMatrix, nShift::Integer)
    if isodd(k)
        mtxlw .= circshift(mtxlw, (0,  nShift))
    else
        mtxup .= circshift(mtxup, (0, -nShift))
    end
    nothing
end

adjshiftcoefs!(v::Val{:circular}, k, mtxup, mtxlw, nShift::Integer) = shiftcoefs!(v, k, mtxup, mtxlw, -nShift)

function shiftcoefs!(::Val{:zero}, k::Integer, mtxup::AbstractMatrix, mtxlw::AbstractMatrix, nShift::Integer)
    if isodd(k)
        mtxlw[:, 1+nShift:end] .= @view mtxlw[:, 1:end-nShift]
        mtxlw[:, 1:nShift] .= 0
    else
        mtxup[:, 1:end-nShift] .= @view mtxup[:, 1+nShift:end]
        mtxup[:, end-nShift+1:end] .= 0
    end
    nothing
end

function adjshiftcoefs!(::Val{:zero}, k::Integer, mtxup::AbstractMatrix, mtxlw::AbstractMatrix, nShift::Integer)
    if isodd(k)
        mtxlw[:, 1:end-nShift] .= @view mtxlw[:, 1+nShift:end]
        mtxlw[:, end-nShift+1:end] .= 0
    else
        mtxup[:, 1+nShift:end] .= @view mtxup[:, 1:end-nShift]
        mtxup[:, 1:nShift] .= 0
    end
    nothing
end

function shiftcoefs!(::Val{:circular_traditional}, ::Integer, ::Any, mtxlw::AbstractMatrix, nShift::Integer)
    mtxlw .= circshift(mtxlw, (0, nShift))
end

adjshiftcoefs!(v::Val{:circular_traditional}, k, mtxup, mtxlw, nShift::Integer) = shiftcoefs!(v, k, mtxup, mtxlw, -nShift)
