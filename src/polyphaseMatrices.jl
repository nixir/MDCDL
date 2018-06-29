
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

function getAnalysisFilters(pfb::MDCDL.ParallelFB)
    pfb.analysisFilters
end

function getSynthesisFilters(pfb::MDCDL.ParallelFB)
    pfb.synthesisFilters
end
