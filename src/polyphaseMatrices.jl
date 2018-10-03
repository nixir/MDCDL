using TiledIteration: TileIterator
using FFTW: fft, dct

Base.@pure function get_rnsolt_default_matrices(::TypeI, ::Type{T}, ppo::NTuple{D}, nChs::Tuple{Int,Int}) where {T,D}
    initMts = Matrix{T}[ Matrix(I, p, p) for p in nChs ]
    propMts = Vector{Matrix{T}}[
        [
            (iseven(n) ? 1 : -1) .* Matrix(I, nChs[1], nChs[1])
        for n in 1:ppo[pd] ]
    for pd in 1:D ]

    (initMts, propMts)
end

Base.@pure function get_rnsolt_default_matrices(::TypeII, ::Type{T}, ppo::NTuple{D}, nChs::Tuple{Int,Int}) where {T,D}
    initMts = Matrix{T}[ Matrix(I, p, p) for p in nChs ]
    chx, chn = maximum(nChs), minimum(nChs)
    propMts = Vector{Matrix{T}}[
        vcat(
            fill([ -Matrix(I, chn, chn), Matrix(I, chx, chx) ], fld(ppo[pd],2))...
        )
    for pd in 1:D ]

    (initMts, propMts)
end

Base.@pure function get_cnsolt_default_matrices(::TypeI, ::Type{T}, ppo::NTuple{D}, nChs::Integer) where {T,D}
    initMts = Matrix{T}[ Matrix(I,nChs,nChs) ]
    propMts = Vector{Matrix{T}}[
        [
            (iseven(n) ? -1 : 1) * Matrix(I,fld(nChs,2),fld(nChs,2))
        for n in 1:2*ppo[pd] ]
    for pd in 1:D ]

    (initMts, propMts)
end

Base.@pure function get_cnsolt_default_matrices(::TypeII, ::Type{T}, ppo::NTuple{D}, nChs::Integer) where {T,D}
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

    (initMts, propMts)
end

# upsampler
function upsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D} = (fill(0,D)...,)) where {T,D}
    szx = size(x)
    mtx = zeros(T, szx .* factor)
    for ci in CartesianIndices(szx)
        mtx[((ci.I .- 1) .* factor .+ 1 .+ offset)...] = x[ci]
    end
    mtx
end

function downsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D}=(fill(0,D)...,)) where {T,D}
    map(CartesianIndices(fld.(size(x), factor))) do ci
        x[((ci.I .- 1) .* factor .+ 1 .+ offset)...]
    end
end

representationmatrix(f, sz::NTuple) = representationmatrix(f, sz...)
function representationmatrix(f::Function, sz::Integer...)
    hcat([ setindex!(zeros(sz), 1, idx) |> f |> vec for idx in 1:prod(sz) ]...)
end

Base.@pure function haarbasis(d::Integer)
    w = walsh(d)
    [ reshape(w[p,:], fill(2,d)...) |> Array for p in 1:size(w, 1)]
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
cdftmtx(::Type{Complex{T}}, sz...) where {T} = cdftmtx(T, sz...)

Base.@pure function cdftmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    len = prod(sz)

    mtx = representationmatrix(x->fft(T.(x)), sz)
    rm = Diagonal(Complex{T}[ exp(-1im*angle(mtx[n,end])/2) for n in 1:len ])

    rm * mtx / sqrt(T(len))
end

permdctmtx(sz::NTuple) = permdctmtx(sz...)
permdctmtx(sz::Integer...) = permdctmtx(Float64, sz...)
permdctmtx(T::Type, sz::NTuple) = permdctmtx(T, sz...)

Base.@pure function permdctmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    mtx = representationmatrix(x->dct(T.(x)), sz)

    isevenids = map(ci->iseven(sum(ci.I .- 1)), CartesianIndices(sz)) |> vec
    permids = sortperm(isevenids; rev=true, alg=Base.DEFAULT_STABLE)

    vcat([ transpose(mtx[pi,:]) for pi in permids ]...)
end


function getMatrixB(P::Integer, angs::AbstractVector{T}) where T
    hP = fld(P,2)
    psangs = (2 .* angs .+ pi) ./ 4
    cs = cos.(psangs)
    ss = sin.(psangs)

    LC = [[ (-1im*cs[n]) (-1im*ss[n]); (cs[n]) (-ss[n]) ] for n in 1:fld(hP,2) ]
    LS = [[ (ss[n]) (cs[n]); (1im*ss[n]) (-1im*cs[n]) ] for n in 1:fld(hP,2) ]

    C, S = if iseven(hP)
        (cat(LC...; dims=[1,2]), cat(LS...; dims=[1,2]))
    else
        (cat(LC...,1; dims=[1,2]), cat(LS...,1im; dims=[1,2]))
    end

    [ C conj(C); S conj(S) ] / sqrt(convert(T,2))
end

analysisbank(cc::AbstractNsolt) = analysisbank(Val(istype1(cc)), cc)

function analysisbank(::TypeI, cc::Cnsolt{T,D}) where {D,T}
    df = cc.decimationFactor
    P = cc.nChannels
    M = prod(df)
    ord = cc.polyphaseOrder

    rngUpper = (1:fld(P,2), :)
    rngLower = (fld(P,2)+1:P, :)

    # output
    ppm = zeros(complex(T), P, prod(df .* (ord .+ 1)))
    ppm[1:M,1:M] = cc.matrixF

    # Initial matrix process
    ppm = cc.initMatrices[1] * ppm

    nStride = M
    for d = 1:D
        angs = cc.paramAngles[d]
        propMats = cc.propMatrices[d]
        for k = 1:ord[d]
            B = getMatrixB(P, angs[k])
            W = propMats[2k-1]
            U = propMats[2k]

            # B Λ(z_d) B'
            ppm = B' * ppm
            ppm[rngLower...] = circshift(ppm[rngLower...],(0, nStride))
            ppm = B * ppm

            ppm[rngUpper...] = W * ppm[rngUpper...]
            ppm[rngLower...] = U * ppm[rngLower...]
        end
        nStride *= ord[d] + 1
    end
    cc.symmetry * ppm
end

function analysisbank(::TypeII, cc::Cnsolt{T,D}) where {D,T}
    df = cc.decimationFactor
    P = cc.nChannels
    M = prod(df)
    ord = cc.polyphaseOrder
    nStages = fld.(ord,2)
    chEven = 1:P-1

    # output
    ppm = zeros(complex(T), P, prod(df .* (ord .+ 1)))
    ppm[1:M,1:M] = cc.matrixF

    # Initial matrix process
    ppm = cc.initMatrices[1] * ppm

    nStride = M
    for d = 1:D
        angs = cc.paramAngles[d]
        propMats = cc.propMatrices[d]
        for k = 1:nStages[d]
            # first step
            chUpper = 1:fld(P,2)
            chLower = fld(P,2)+1:P-1
            B = getMatrixB(P, angs[2k-1])
            W = propMats[4k-3]
            U = propMats[4k-2]

            # B Λ(z_d) B'
            ppm[chEven,:] = B' * ppm[chEven,:]
            ppm[chLower,:] = circshift(ppm[chLower,:],(0, nStride))
            ppm[chEven,:] = B * ppm[chEven,:]

            ppm[chUpper,:] = W * ppm[chUpper,:]
            ppm[chLower,:] = U * ppm[chLower,:]

            # second step
            chUpper = 1:cld(P,2)
            chLower = cld(P,2):P

            B = getMatrixB(P, angs[2k])
            hW = propMats[4k-1]
            hU = propMats[4k]

            # B Λ(z_d) B'
            ppm[chEven,:] = B' * ppm[chEven,:]
            ppm[chLower,:] = circshift(ppm[chLower,:],(0, nStride))
            ppm[chEven,:] = B * ppm[chEven,:]

            ppm[chLower,:] = hU * ppm[chLower,:]
            ppm[chUpper,:] = hW * ppm[chUpper,:]
        end
        nStride *= ord[d] + 1
    end
    cc.symmetry * ppm
end

function analysisbank(::TypeI, rc::Rnsolt{T,D}) where {D,T}
    df = rc.decimationFactor
    nch = rc.nChannels
    P = sum(nch)
    M = prod(df)
    ord = rc.polyphaseOrder

    rngUpper = (1:nch[1], :)
    rngLower = (nch[1]+1:P, :)

    # output
    ppm = zeros(T, P, prod(df .* (ord .+ 1)))
    ppm[1:cld(M,2), 1:M] = rc.matrixC[1:cld(M,2),:]
    ppm[nch[1].+(1:fld(M,2)), 1:M] = rc.matrixC[cld(M,2)+1:end,:]

    # Initial matrix process
    ppm[rngUpper...] = rc.initMatrices[1] * ppm[rngUpper...]
    ppm[rngLower...] = rc.initMatrices[2] * ppm[rngLower...]

    nStride = M
    for d = 1:D
        propMats = rc.propMatrices[d]
        for k = 1:ord[d]
            U = propMats[k]

            # B Λ(z_d) B'
            butterfly!(ppm, nch[1])
            ppm[rngLower...] = circshift(ppm[rngLower...],(0, nStride))
            butterfly!(ppm, nch[1])

            ppm[rngLower...] = U * ppm[rngLower...]
        end
        nStride *= ord[d] + 1
    end
    ppm
end

function analysisbank(::TypeII, rc::Rnsolt{T,D}) where {D,T}
    df = rc.decimationFactor
    M = prod(df)
    ord = rc.polyphaseOrder
    nStages = fld.(rc.polyphaseOrder,2)
    nch = rc.nChannels
    P = sum(nch)
    maxP, minP, chMajor, chMinor = if rc.nChannels[1] > rc.nChannels[2]
        (rc.nChannels[1], rc.nChannels[2], 1:rc.nChannels[1], (rc.nChannels[1]+1):P)
    else
        (rc.nChannels[2], rc.nChannels[1], (rc.nChannels[1]+1):P, 1:rc.nChannels[1])
    end

    # output
    ppm = zeros(T, P, prod(df .* (ord .+ 1)))
    # ppm[1:M,1:M] = rc.matrixF
    ppm[1:cld(M,2), 1:M] = rc.matrixC[1:cld(M,2),:]
    ppm[nch[1] .+ (1:fld(M,2)), 1:M] = rc.matrixC[cld(M,2)+1:end,:]

    # Initial matrix process
    ppm[1:nch[1],:] = rc.initMatrices[1] * ppm[1:nch[1],:]
    ppm[(nch[1]+1):end,:] = rc.initMatrices[2] * ppm[(nch[1]+1):end,:]

    nStride = M
    for d = 1:D
        propMats = rc.propMatrices[d]
        for k = 1:nStages[d]
            # first step

            U = propMats[2k-1]

            # B Λ(z_d) B'
            butterfly!(ppm, minP)
            ppm[(minP+1):end,:] = circshift(ppm[(minP+1):end,:], (0, nStride))
            butterfly!(ppm, minP)

            ppm[chMinor,:] = U * ppm[chMinor,:]

            # second step
            W = propMats[2k]

            # B Λ(z_d) B'
            butterfly!(ppm, minP)
            ppm[(maxP+1):end,:] = circshift(ppm[(maxP+1):end,:], (0, nStride))
            butterfly!(ppm, minP)

            ppm[chMajor,:] = W * ppm[chMajor,:]
        end
        nStride *= ord[d] + 1
    end
    ppm
end

kernels(pfb::PolyphaseFB) = (analysiskernels(pfb), synthesiskernels(pfb))

function analysiskernels(pfb::PolyphaseFB)
    df = decimations(pfb)
    P = nchannels(pfb)

    afb = analysisbank(pfb)
    ordm = orders(pfb) .+ 1

    return map(1:P) do p
        out = similar(afb, df .* ordm )
        tilesout = collect(TileIterator(axes(out), df))

        for idx in LinearIndices(ordm)
            sub = (1:prod(df)) .+ ((idx - 1) * prod(df))
            out[tilesout[idx]...] = reshape(afb[p, sub], df...)
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
    if any(size(x) .% szBlock .!= 0)
        error("size error. input data: $(size(x)), block size: $(szBlock).")
    end

    outdata = similar(x, prod(szBlock), prod(nBlocks))
    tiles = collect(TileIterator(axes(x), szBlock))
    for idx in 1:length(tiles)
        outdata[:,idx] = vec(x[tiles[idx]...])
    end
    PolyphaseVector(outdata, nBlocks)
end

function mdarray2polyphase(x::AbstractArray{T,D}) where {T,D}
    nBlocks = size(x)[1:D-1]
    outdata = transpose(reshape(x, prod(nBlocks), size(x,D)))
    PolyphaseVector(outdata, nBlocks)
end

function polyphase2mdarray(x::PolyphaseVector{TX,D}, szBlock::NTuple{D,TS}) where {TX,D,TS<:Integer}
    if size(x.data,1) != prod(szBlock)
        throw(ArgumentError("size mismatch! 'prod(szBlock)' must be equal to $(size(x.data,1))."))
    end

    out = similar(x.data, (x.nBlocks .* szBlock)...)
    tiles = collect(TileIterator(axes(out), szBlock))
    for idx in 1:length(tiles)
        out[tiles[idx]...] = reshape(x.data[:,idx], szBlock...)
    end
    out
end

function polyphase2mdarray(x::PolyphaseVector{T,D}) where {T,D}
    reshape(transpose(x.data), x.nBlocks..., size(x.data, 1))
end

function permutedimspv(x::AbstractMatrix, nShift::Integer)
    S = fld(size(x,2), nShift)
    data = similar(x)
    for idx = 0:nShift - 1
        data[:,(1:S) .+ idx*S] = x[:, (1:nShift:end) .+ idx]
    end
    data
end

function ipermutedimspv(x::AbstractMatrix, nShift::Integer)
    S = fld(size(x, 2), nShift)
    data = similar(x)
    for idx = 0:S-1
        data[:,(1:nShift) .+ idx*nShift] = x[:, (1:S:end) .+ idx]
    end
    data
end

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

function shiftforward!(::Val{:circular}, mtx::AbstractMatrix, nShift::Integer)
    mtx .= circshift(mtx, (0, nShift))
end
shiftbackward!(tp::Val{:circular}, mtx, nShift) = shiftforward!(tp, mtx, -nShift)

function shiftforward!(::Val{:zero}, mtx::AbstractMatrix, nShift::Integer)
    mtx[:,1+nShift:end] .= mtx[:,1:end-nShift]
    mtx[:,1:nShift] .= 0
    mtx
end

function shiftbackward!(::Val{:zero}, mtx::AbstractMatrix, nShift::Integer)
    mtx[:,1:end-nShift] .= mtx[:,1+nShift:end]
    mtx[:,end-nShift+1:end] .= 0
    mtx
end

shiftforward(tp::Val, mtx::AbstractMatrix, nShift) = shiftforward!(tp, deepcopy(mtx), nShift)

shiftbackward(tp::Val, mtx::AbstractMatrix, nShift) = shiftbackward!(tp, deepcopy(mtx), nShift)
