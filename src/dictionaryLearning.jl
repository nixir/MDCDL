
function optimizeDictionary(cb::CodeBook, trainingSet::AbstractArray)

end

# grad_{θ}(1/2 *|| x - D(θ)y ||_2^2)
# D: RNSOLT synthesizer
function gradSqrdError(nsolt::Rnsolt{T,D,:TypeI}, x::PolyphaseVector{T,D}, y::PolyphaseVector{T,D}) where {T,D}
    df = nsolt.decimationFactor
    ord = nsolt.polyphaseOrder
    nch = nsolt.nChannels

    rpvx = synthesize(nsolt, y)
    rx = rpvx.data
    nBlocks = rpvx.nBlocks

    # error vector
    et = mdarray2polyphase(x - rx, df).data

    yt = mdarray2polyphase(rx, df).data

    M = prod(df)
    cM, fM = cld(M,2), fld(M,2)

    Pu, Pl = eye(T, nch[1], cM), eye(T, nch[2], fM)
    P = zeros(T, sum(nch), M)
    P[1:nch[1],1:cM] = Pu
    P[nch[1]+1:end,cM+1:end] = Pl

    pet = P * nsolt.matrixC * et
    pyt = P * nsolt.matrixC * yt

    W0 = nsolt.initMatrices[1]
    U0 = nsolt.initMatrices[2]

    pyt[1:nch[1],:] = W0 * pyt[1:nch[1],:]
    pyt[(nch[1]+1):end,:] = U0 * pyt[(nch[1]+1):end,:]

    angsw, musw = mat2rotations(W0)
    angsu, musu = mat2rotations(U0)

    gdw = - scalarGradOfOrthonormalMatrix(pyt[1:nch[1],:], pet[1:nch[1],:], angsw, musw)
    gdu = - scalarGradOfOrthonormalMatrix(pyt[nch[1]+1:end,:], pet[nch[1]+1:end,:], angsu, musu)

    pet[1:nch[1],:] = W0*pet[1:nch[1],:]
    pet[nch[1]+1:end,:] = U0*pet[nch[1]+1:end,:]

    hP = nsolt.nChannels[1]

    L = fld(nch[2]*(nch[2]-1),2)
    gdudk = [ fill(zeros(T,L), o) for o in ord ]
    for d = 1:D
        nShift = fld(size(pet,2), nBlocks[1])
        # pvx = permutedims(pvx)
        pvxe = MDCDL.permutedims(PolyphaseVector(pet, nBlocks))
        pet = pvxe.data
        pvy  = MDCDL.permutedims(PolyphaseVector(pyt, nBlocks))
        pyt, nBlocks = pvy.data, pvy.nBlocks
        # submatrices
        xeu = view(pet, 1:hP, :)
        xel = view(pet, (1:hP)+hP, :)
        xyu = view(pyt, 1:hP, :)
        xyl = view(pyt, (1:hP)+hP, :)
        for k = 1:nsolt.polyphaseOrder[d]
            teu, tel = (xeu + xel, xeu - xel) ./ sqrt(2)
            xeu .= teu; xel .= tel

            tyu, tyl = (xyu + xyl, xyu - xyl) ./ sqrt(2)
            xyu .= tyu; xyl .= tyl

            if isodd(k)
                xel .= circshift(xel, (0, nShift))
                xyl .= circshift(xyl, (0, nShift))
            else
                xeu .= circshift(xeu, (0, -nShift))
                xyu .= circshift(xyu, (0, -nShift))
            end

            teu, tel = (xeu + xel, xeu - xel) ./ sqrt(2)
            xeu .= teu; xel .= tel

            tyu, tyl = (xyu + xyl, xyu - xyl) ./ sqrt(2)
            xyu .= tyu; xyl .= tyl

            Uk = nsolt.propMatrices[d][k]
            xyl .= Uk * xyl

            angsu, musu = mat2rotations(nsolt.propMatrices[d][k])
            gdudk[d][k] = - scalarGradOfOrthonormalMatrix(xyl, xel, angsu, musu)

            xel .= Uk * xel
        end
    end

    vcat(gdw[nch[1]:end], gdu, vcat(vcat.(gdudk...)...))
end
