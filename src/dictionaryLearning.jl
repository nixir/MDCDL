
function optimizeDictionary(cb::CodeBook, trainingSet::AbstractArray)

end

# grad_{θ}(1/2 *|| x - D(θ)y ||_2^2)
# D: RNSOLT synthesizer
function gradSqrdError(nsolt, x, y)
    errvec = PolyphaseVector(x.data - synthesize(nsolt, y).data, x.nBlocks)
    return -gradOfAnalyzer(nsolt, errvec, y)
end

function gradOfAnalyzer(nsolt::Rnsolt{T,D,:TypeI}, x::PolyphaseVector{T,D}, y::PolyphaseVector{T,D}) where {T,D}
    df = nsolt.decimationFactor
    ord = nsolt.polyphaseOrder
    nch = nsolt.nChannels

    rpvy = analyze(nsolt, x) # D^T * x
    nBlocks = rpvy.nBlocks

    rt = rpvy.data
    yt = copy(y.data)

    M = prod(df)
    cM, fM = cld(M,2), fld(M,2)

    hP = nsolt.nChannels[1]

    L = fld(nch[2]*(nch[2]-1),2)
    gdudk = [ fill(zeros(T,L), o) for o in ord ]
    for d = D:-1:1
        nShift = fld(size(rt,2), nBlocks[end])
        # submatrices
        yu = view(yt, 1:hP, :)
        yl = view(yt, (1:hP)+hP, :)
        ru = view(rt, 1:hP, :)
        rl = view(rt, (1:hP)+hP, :)
        for k = nsolt.polyphaseOrder[d]:-1:1
            Uk = nsolt.propMatrices[d][k]
            rl .= Uk' * rl
            gdudk[d][k] = MDCDL.scalarGradOfOrthonormalMatrix(yl, rl, Uk)
            yl .= Uk' * yl

            tyu, tyl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tyu; yl .= tyl
            tru, trl = (ru + rl, ru - rl) ./ sqrt(2)
            ru .= tru; rl .= trl
            if isodd(k)
                yl .= circshift(yl, (0, -nShift))
                rl .= circshift(rl, (0, -nShift))
            else
                yu .= circshift(yu, (0, nShift))
                ru .= circshift(ru, (0, nShift))
            end
            tyu, tyl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tyu; yl .= tyl
            tru, trl = (ru + rl, ru - rl) ./ sqrt(2)
            ru .= tru; rl .= trl
        end
        pvyt = MDCDL.ipermutedims(PolyphaseVector(yt, nBlocks))
        pvrt = MDCDL.ipermutedims(PolyphaseVector(rt, nBlocks))
        yt = pvyt.data
        rt = pvrt.data
        nBlocks = pvrt.nBlocks
    end

    W0om = nsolt.initMatrices[1]
    U0om = nsolt.initMatrices[2]
    Iw = eye(T, nch[1], cM)
    Iu = eye(T, nch[2], fM)
    W0 = W0om * Iw
    U0 = U0om * Iu
    rts = vcat(W0' * rt[1:nch[1],:], U0' * rt[nch[1]+1:end,:])
    gdw = MDCDL.scalarGradOfOrthonormalMatrix(yt[1:nch[1],:], Iw * rts[1:cM,:], W0om)
    gdu = MDCDL.scalarGradOfOrthonormalMatrix(yt[nch[1]+1:end,:], Iu * rts[cM+1:end,:], U0om)
    # yts = vcat(W0' * yt[1:nch[1],:], U0' * yt[nch[1]+1:end,:])
    #
    # rts .= nsolt.matrixC' * rts
    # yts .= nsolt.matrixC' * yts

    vcat(gdw, gdu, vcat(vcat.(gdudk...)...))
end
