using TiledIteration: TileIterator

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

analysisbank(cc::AbstractNsolt) = analysisbank(Val{cc.category}, cc)

function analysisbank(::Type{Val{:TypeI}}, cc::Cnsolt{T,D}) where {D,T}
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

function analysisbank(::Type{Val{:TypeII}}, cc::Cnsolt{T,D}) where {D,T}
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

function analysisbank(::Type{Val{:TypeI}}, rc::Rnsolt{T,D}) where {D,T}
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

function analysisbank(::Type{Val{:TypeII}}, rc::Rnsolt{T,D}) where {D,T}
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
    df = pfb.decimationFactor
    P = sum(pfb.nChannels)

    afb = analysisbank(pfb)
    ordm = pfb.polyphaseOrder .+ 1

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
    for idx in LinearIndices(nBlocks)
        outdata[:,idx] = vec(x[tiles[idx]...])
    end
    PolyphaseVector(outdata, nBlocks)
end

function mdarray2polyphase(x::AbstractArray{T,D}) where {T,D}
    nBlocks = size(x)[1:D-1]
    outdata = similar(x, size(x,D), prod(nBlocks))
    for p = 1:size(x,D)
        outdata[p,:] = vec( x[fill(:,D-1)..., p])
    end
    PolyphaseVector(outdata, nBlocks)
end

function polyphase2mdarray(x::PolyphaseVector{TX,D}, szBlock::NTuple{D,TS}) where {TX,D,TS<:Integer}
    if size(x.data,1) != prod(szBlock)
        throw(ArgumentError("size mismatch! 'prod(szBlock)' must be equal to $(size(x.data,1))."))
    end

    out = similar(x.data, (x.nBlocks .* szBlock)...)
    tiles = collect(TileIterator(axes(out), szBlock))
    for idx in LinearIndices(x.nBlocks)
        out[tiles[idx]...] = reshape(x.data[:,idx], szBlock...)
    end
    out
end

function polyphase2mdarray(x::PolyphaseVector{T,D}) where {T,D}
    P = size(x.data,1)
    output = similar(x.data, x.nBlocks..., P)

    for p = 1:P
        output[fill(:,D)...,p] = reshape(x.data[p,:], x.nBlocks)
    end
    output
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

function shiftforward!(::Type{Val{:circular}}, mtx::AbstractMatrix, nShift::Integer)
    mtx .= circshift(mtx, (0, nShift))
end
shiftbackward!(::Type{Val{:circular}}, mtx, nShift) = shiftforward!(Val{:circular}, mtx, -nShift)

function shiftforward!(::Type{Val{:zero}}, mtx::AbstractMatrix, nShift::Integer)
    mtx[:,1+nShift:end] .= mtx[:,1:end-nShift]
    mtx[:,1:nShift] .= 0
    mtx
end

function shiftbackward!(::Type{Val{:zero}}, mtx::AbstractMatrix, nShift::Integer)
    mtx[:,1:end-nShift] .= mtx[:,1+nShift:end]
    mtx[:,end-nShift+1:end] .= 0
    mtx
end

shiftforward(tp::Type{T}, mtx::AbstractMatrix, nShift) where {T} = shiftforward!(tp, deepcopy(mtx), nShift)

shiftbackward(tp::Type{T}, mtx::AbstractMatrix, nShift) where {T} = shiftbackward!(tp, deepcopy(mtx), nShift)
