import Base.convert
import Base.*
import Base.size
import Base.getindex
import Base.setindex!
# import Base.=

function getMatrixB(P::Integer, angs::Vector{T}) where T
    hP = fld(P,2)
    psangs = (2 .* angs .+ pi) ./ 4
    cs = cos.(psangs)
    ss = sin.(psangs)

    subMatFcn = (x) -> sparse([1,1,2,2], [1,2,1,2], x)

    LC = [ subMatFcn([ -1im*cs[n], -1im*ss[n], cs[n], -ss[n] ]) for n in 1:fld(hP,2) ]
    LS = [ subMatFcn([ ss[n], cs[n], 1im*ss[n], -1im*cs[n] ]) for n in 1:fld(hP,2) ]

    C, S = if hP % 2 == 0
        (Array(blkdiag(LC...)), Array(blkdiag(LS...)))
    else
        (Array(blkdiag(LC...,sparse([1],[1],[1]))), Array(blkdiag(LS...,sparse([1],[1],[1im]))))
    end

    [ C conj(C); S conj(S) ] / sqrt(convert(T,2))
end

function getAnalysisBank(cc::MDCDL.Cnsolt{D,1,T}) where {D,T}
    df = cc.decimationFactor
    P = cc.nChannels
    M = prod(df)
    ord = cc.polyphaseOrder

    rngUpper = (1:fld(P,2), :)
    rngLower = (fld(P,2)+1:P, :)

    # output
    ppm = zeros(Complex{T},P, prod(df .* (ord .+ 1)))
    ppm[1:M,1:M] = cc.matrixF

    # Initial matrix process
    ppm = cc.initMatrices[1] * ppm

    nStride = M
    for d = 1:D
        angs = cc.paramAngles[d]
        propMats = cc.propMatrices[d]
        for k = 1:ord[d]
            B = MDCDL.getMatrixB(P, angs[k])
            W = propMats[2*k-1]
            U = propMats[2*k]

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

function getAnalysisBank(rc::MDCDL.Rnsolt{D,1,T}) where {D,T}
    df = rc.decimationFactor
    nch = rc.nChannels
    P = sum(nch)
    M = prod(df)
    ord = rc.polyphaseOrder

    rngUpper = (1:nch[1], :)
    rngLower = (nch[1]+1:P, :)

    # output
    ppm = zeros(T, P, prod(df .* (ord .+ 1)))
    # ppm[1:M,1:M] = rc.matrixF
    ppm[1:cld(M,2), 1:M] = rc.matrixC[1:cld(M,2),:]
    ppm[nch[1]+(1:fld(M,2)), 1:M] = rc.matrixC[cld(M,2)+1:end,:]

    # Initial matrix process
    ppm[rngUpper...] = rc.initMatrices[1] * ppm[rngUpper...]
    ppm[rngLower...] = rc.initMatrices[2] * ppm[rngLower...]

    nStride = M
    for d = 1:D
        propMats = rc.propMatrices[d]
        for k = 1:ord[d]
            U = propMats[k]

            # B Λ(z_d) B'
            ppm = MDCDL.butterfly(ppm, nch[1])
            ppm[rngLower...] = circshift(ppm[rngLower...],(0, nStride))
            ppm = MDCDL.butterfly(ppm, nch[1])

            ppm[rngLower...] = U * ppm[rngLower...]
        end
        nStride *= ord[d] + 1
    end
    ppm
end

function getAnalysisFilters(pfb::MDCDL.PolyphaseFB{T,D}) where {T,D}
    df = pfb.decimationFactor
    P = sum(pfb.nChannels)

    afb = MDCDL.getAnalysisBank(pfb)
    # primeBlock = ntuple(d -> 1:df[d], D)
    primeBlock = colon.(1,df)
    ordm = pfb.polyphaseOrder .+ 1
    vecfs = [ afb[p,:] for p in 1:P ]

    return map(1:P) do p
        out = Array{T}(df .* ordm )

        foreach(1:prod(ordm)) do idx
            sub = ind2sub(ordm, idx)
            subaf = primeBlock .+ (sub .- 1) .* df
            subfb = (1:prod(df)) + (idx-1) .* prod(df)

            out[subaf...] = reshape(afb[ p, subfb ], df...)
        end
        out
    end
end

function getSynthesisFilters(cc::MDCDL.Cnsolt)
    map(MDCDL.getAnalysisFilters(cc)) do af
        sz = size(af)
        reshape(flipdim(vec(conj.(af)),1),sz)
    end
end

function getSynthesisFilters(rc::MDCDL.Rnsolt)
    map(MDCDL.getAnalysisFilters(rc)) do af
        sz = size(af)
        reshape(flipdim(vec(af),1),sz)
    end
end

# function convert(::Type{Array}, x::PolyphaseVector{T,D}) where {T,D}
#     primeBlock = colon.(1, x.szBlock)
#     output = Array{T,D}((x.szBlock .* x.nBlocks)...)
#     foreach(1:prod(x.nBlocks)) do idx
#         block = (ind2sub(x.nBlocks, idx) .- 1) .* x.szBlock .+ primeBlock
#         output[block...] = reshape(x.data[:,idx], x.szBlock...)
#     end
#     output
# end

function mdarray2polyphase(x::Array{TX,D}, szBlock::NTuple{D,TS}) where {TX,D,TS<:Integer}
    nBlocks = fld.(size(x), szBlock)
    if any(size(x) .% szBlock .!= 0)
        error("size error. input data: $(size(x)), block size: $(szBlock).")
    end
    primeBlock = colon.(1, szBlock)

    data = hcat([ vec( x[ ((ind2sub(nBlocks,idx) .- 1) .* szBlock .+ primeBlock)... ]) for idx in 1:prod(nBlocks) ]...)
    PolyphaseVector(data, nBlocks)
end

function mdarray2polyphase(x::Array{T,D}) where {T,D}
    data = vcat([ vec(x[fill(:,D-1)...,p]).' for p in 1:size(x,D) ]...)
    nBlocks = size(x)[1:end-1]
    PolyphaseVector(data, nBlocks)
end

function polyphase2mdarray(x::PolyphaseVector{TX,D}, szBlock::NTuple{D,TS}) where {TX,D,TS<:Integer}
    if size(x.data,1) != prod(szBlock)
        error("size mismatch! 'prod(szBlock)' must be equal to $(size(x.data,1)).")
    end

    # primeBlock = ntuple(n -> 1:szBlock[n], D)
    primeBlock = colon.(1, szBlock)
    out = Array{TX,D}((x.nBlocks .* szBlock)...)
    foreach(1:prod(x.nBlocks)) do idx
        out[((ind2sub(x.nBlocks,idx) .- 1) .* szBlock .+ primeBlock)...] = reshape(x.data[:,idx],szBlock...)
    end
    out
end

function polyphase2mdarray(x::PolyphaseVector{T,D}) where {T,D}
    const P = size(x.data,1)
    output = Array{T,D+1}(x.nBlocks..., P)

    foreach(1:P) do p
        output[fill(:,D)...,p] = reshape(x.data[p,:], x.nBlocks)
    end
    output
end

function permutedims(x::PolyphaseVector{T,D}) where {T,D}
    data = hcat( [ x.data[:, (1:x.nBlocks[1]:end) + idx] for idx in 0:x.nBlocks[1]-1 ]... )
    nBlocks = tuple(circshift(collect(x.nBlocks),-1)...)

    PolyphaseVector(data,nBlocks)
end

function ipermutedims(x::PolyphaseVector{T,D}) where {T,D}
    S = fld(size(x.data,2), x.nBlocks[1])
    data = hcat( [ x.data[:, (1:S:end) + idx] for idx in 0:S-1 ]... )
    nBlocks = tuple(circshift(collect(x.nBlocks),1)...)

    PolyphaseVector(data,nBlocks)
end

multiple(mtx::Matrix, pv::PolyphaseVector{T,D}) where {T,D} = PolyphaseVector{T,D}(mtx*pv.data, pv.nBlocks)

*(mtx::Matrix, pv::PolyphaseVector{T,D}) where {T,D} = PolyphaseVector(mtx*pv.data, pv.nBlocks)
*(mtx::Diagonal, pv::PolyphaseVector{T,D}) where {T,D} = PolyphaseVector(mtx*pv.data, pv.nBlocks)

size(A::PolyphaseVector) = size(A.data)

getindex(A::PolyphaseVector, i::Int) = getindex(A.data, i)

getindex(A::PolyphaseVector, I::Vararg{Int, 2}) = getindex(A.data, I...)

function setindex!(A::PolyphaseVector, v, i::Int)
    setindex!(A.data, v, i)
    A
end

function setindex!(A::PolyphaseVector, v, I::Vararg{Int, 2})
    setindex!(A.data, v, I...)
    A
end

copy(A::PolyphaseVector{T,D}) where {T,D} = PolyphaseVector(A.data, A.nBlocks)
