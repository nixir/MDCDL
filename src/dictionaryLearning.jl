
function train!(cb::CodeBook, trainingSet::AbstractArray; epochs=1, verbose=:none)
    nothing
end

# grad_{θ}(1/2 *|| x - D(θ)y ||_2^2)
# D: RNSOLT synthesizer
function gradSqrdError(nsolt::Nsolt, x, y)
    errvec = PolyphaseVector(x.data - synthesize(nsolt, y).data, x.nBlocks)
    return -1 .* real.(gradOfAnalyzer(nsolt, errvec, y))
end

function gradOfAnalyzer(nsolt::Cnsolt{T,D,:TypeI}, x0::PolyphaseVector{TV,D}, y0::PolyphaseVector{TV,D}; border=:circular) where {T,TV,D}

    # error("this method hasn't implemented yet.")

    df = nsolt.decimationFactor
    ord = nsolt.polyphaseOrder
    nch = nsolt.nChannels

    rpvy = analyze(nsolt, x0) # D^T * x
    nBlocks = rpvy.nBlocks

    rt = nsolt.symmetry' * rpvy.data
    yt = nsolt.symmetry' * copy(y0.data)

    M = prod(df)
    cM, fM = cld(M,2), fld(M,2)

    hP = fld(nch, 2)

    L = fld(hP*(hP-1),2)
    gdudk = [ fill(zeros(Complex{T},L), 2*o) for o in ord ]
    gdθ = [ fill(zeros(Complex{T},fld(nch,4)), o) for o in ord]
    for d = D:-1:1
        nShift = fld(size(rt,2), nBlocks[end])
        # submatrices
        y  = view(yt, :, :)
        yu = view(yt, 1:hP, :)
        yl = view(yt, (1:hP) .+ hP, :)
        r  = view(rt, :, :)
        ru = view(rt, 1:hP, :)
        rl = view(rt, (1:hP) .+ hP, :)
        for k = nsolt.polyphaseOrder[d]:-1:1
            Wk = nsolt.propMatrices[d][2k-1]
            Uk = nsolt.propMatrices[d][2k]
            ru .= Wk' * ru
            rl .= Uk' * rl
            gdudk[d][2k-1] = MDCDL.scalarGradOfOrthonormalMatrix(yu, ru, Wk)
            gdudk[d][2k  ] = MDCDL.scalarGradOfOrthonormalMatrix(yl, rl, Uk)
            yu .= Wk' * ru
            yl .= Uk' * yl

            angs = nsolt.paramAngles[d][k]
            B = getMatrixB(nch, angs)
            r .= B' * r
            if isodd(k)
                shiftBackward!(Val{border}, rl, nShift)
            else
                shiftForward!(Val{border}, ru, nShift)
            end
            r .= B * r
            #************ Processing *************
            gdθ[d][k] = get∇BΛBᵀ(y, r, nch, angs, k, nShift, border)
            #************    End     *************
            y .= B' * y
            if isodd(k)
                shiftBackward!(Val{border}, yl, nShift)
            else
                shiftForward!(Val{border}, yu, nShift)
            end
            y .= B * y
        end
        pvyt = MDCDL.ipermutedims(PolyphaseVector(yt, nBlocks))
        pvrt = MDCDL.ipermutedims(PolyphaseVector(rt, nBlocks))
        yt = pvyt.data
        rt = pvrt.data
        nBlocks = pvrt.nBlocks
    end

    V0om = nsolt.initMatrices[1]
    Iv = Matrix{Complex{T}}(I, nch, M)
    V0 = V0om * Iv
    rts = V0' * rt
    # gdv = MDCDL.scalarGradOfOrthonormalMatrix(yt, Iv * rts, V0om)
    gdv = MDCDL.scalarGradOfOrthonormalMatrix(yt, V0om' * rt, V0om)
    # yts = vcat(W0' * yt[1:nch[1],:], U0' * yt[nch[1]+1:end,:])
    #
    # rts .= nsolt.matrixC' * rts
    # yts .= nsolt.matrixC' * yts

    # return gdv, gdudk, gdθ
    tmp = [ [ vcat(gdudk[d][2k-1], gdudk[d][2k], gdθ[d][k]) for k in 1:ord[d] ] for d in 1:D ]

    vcat(gdv, (vcat.(tmp...)...))
end

function gradOfAnalyzer(nsolt::Rnsolt{T,D,:TypeI}, x::PolyphaseVector{T,D}, y::PolyphaseVector{T,D}; border=:circular) where {T,D}
    df = nsolt.decimationFactor
    ord = nsolt.polyphaseOrder
    nch = nsolt.nChannels

    ∇θ(args...) = MDCDL.scalarGradOfOrthonormalMatrix(args...)

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
        yl = view(yt, (1:hP) .+ hP, :)
        ru = view(rt, 1:hP, :)
        rl = view(rt, (1:hP) .+ hP, :)
        for k = nsolt.polyphaseOrder[d]:-1:1
            Uk = nsolt.propMatrices[d][k]
            rl .= Uk' * rl
            gdudk[d][k] = ∇θ(yl, rl, Uk)
            yl .= Uk' * yl

            tyu, tyl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tyu; yl .= tyl
            tru, trl = (ru + rl, ru - rl) ./ sqrt(2)
            ru .= tru; rl .= trl
            if isodd(k)
                shiftBackward!(Val{border}, yl, nShift)
                shiftBackward!(Val{border}, rl, nShift)
            else
                shiftForward!(Val{border}, yu, nShift)
                shiftForward!(Val{border}, ru, nShift)
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
    Iw = Matrix{T}(I, nch[1], cM)
    Iu = Matrix{T}(I, nch[2], fM)
    W0 = W0om * Iw
    U0 = U0om * Iu
    rts = vcat(W0' * rt[1:nch[1],:], U0' * rt[nch[1]+1:end,:])
    gdw = ∇θ(yt[1:nch[1],:], Iw * rts[1:cM,:], W0om)
    gdu = ∇θ(yt[nch[1]+1:end,:], Iu * rts[cM+1:end,:], U0om)

    # gdw = ∇θ(yt[1:nch[1],:], W0om' * rt[1:nch[1],:], W0om)
    # gdu = ∇θ(yt[nch[1]+1:end,:], U0om' * rt[nch[1]+1:end,:], U0om)
    # yts = vcat(W0' * yt[1:nch[1],:], U0' * yt[nch[1]+1:end,:])
    #
    # rts .= nsolt.matrixC' * rts
    # yts .= nsolt.matrixC' * yts

    vcat(gdw, gdu, vcat(vcat.(gdudk...)...))
end

function get∇BΛBᵀ(x::AbstractArray{TV,D}, y::AbstractArray{TV,D}, P::Integer, θs::AbstractVector{TA}, k::Integer, nShift::Integer, border::Symbol) where {TV,TA,D}
    get∇BΛBᵀ_reference(x, y, P, θs, k, nShift, border)
end

function get∇BΛBᵀ_reference(x::AbstractArray{TV,D}, y::AbstractArray{TV,D}, P::Integer, θs::AbstractVector{TA}, k::Integer, nShift::Integer, border::Symbol) where {TV,TA,D}
    hP = fld(P,2)
    map(1:length(θs)) do idx
        ang = θs[idx] + pi/2
        c, s = cos(ang), sin(ang)

        lc = [ (-1im*c) (-1im*s); (c) (-s) ]
        ls = [ (s) (c); (1im*s) (-1im*c) ]

        ∇C, ∇S = if iseven(hP)
            zeros(complex(TV),fill(hP,2)...), zeros(complex(TV), fill(hP,2)...)
        else
            cat(zeros(complex(TV),fill(hP-1,2)...), 1; dims=[1,2]), cat(zeros(complex(TV),fill(hP-1,2)...), 1im; dims=[1,2])
        end
        ∇C[fill((1:2) .+ 2*(idx-1),2)...] = lc
        ∇S[fill((1:2) .+ 2*(idx-1),2)...] = ls

        ∇B = [ ∇C conj(∇C); ∇S conj(∇S) ] / sqrt(convert(TV,2))

        ty = copy(∇B' * y)
        if isodd(k)
            shiftBackward!(Val{border}, ty[1:hP,:], nShift)
        else
            shiftForward!(Val{border}, ty[hP+1:end,:], nShift)
        end

        dot(x, ∇B * ty)
    end
end
