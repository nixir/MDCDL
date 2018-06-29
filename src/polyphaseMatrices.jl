
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

function getAnalysisFilters(cc::MDCDL.Cnsolt{D,S,T}) where {D,S,T}
    df = cc.decimationFactor

    afb = MDCDL.getAnalysisBank(cc)
    # primeBlock = ntuple(d -> 1:df[d], D)
    primeBlock = colon.(1,df)
    ordm = cc.polyphaseOrder .+ 1
    vecfs = [ afb[p,:] for p in 1:cc.nChannels ]

    return map(1:cc.nChannels) do p
        out = complex(zeros(cc.decimationFactor .* ordm ))

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

function getAnalysisFilters(pfb::MDCDL.ParallelFB)
    pfb.analysisFilters
end

function getSynthesisFilters(pfb::MDCDL.ParallelFB)
    pfb.synthesisFilters
end
