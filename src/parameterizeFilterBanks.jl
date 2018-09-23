
#TODO: 角度パラメータのベクトル化手法の仕様をどこかに記述する．
#TODO: コードが汚いのでリファクタリングする

getrotations(cc::AbstractNsolt) = getrotations(Val(istype1(cc)), cc)
setrotations!(cc::AbstractNsolt, θ, μ) = setrotations!(Val(istype1(cc)), cc, θ, μ)

setrotations(cc::AbstractNsolt, args...) = setrotations!(deepcopy(cc), args...)

function getrotations(::TypeI, cc::Cnsolt{T,D}) where {D,T}
    P = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-1),2)
    nParamsPropPerDimsOrder = fld(P*(P-2),4) + fld(P,4)
    nParamsProps = ord .* nParamsPropPerDimsOrder
    # nParams = vcat(nParamsInit, nParamsProps...)

    # Angles ang MUS
    angsInit, musInit = mat2rotations(cc.initMatrices[1])

    angsPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    for d = 1:D
        angsPropPerDim = Array{Vector{T}}(undef, ord[d])
        musPropPerDim = Array{Vector{T}}(undef, ord[d])
        for k = 1:ord[d]
            (apw, mpw) = mat2rotations(cc.propMatrices[d][2*k-1])
            (apu, mpu) = mat2rotations(cc.propMatrices[d][2*k])
            apb = cc.paramAngles[d][k]

            angsPropPerDim[k] = vcat(apw,apu,apb)
            musPropPerDim[k] = vcat(mpw,mpu)
        end
        angsPropsSet[d] = vcat(angsPropPerDim...)
        musPropsSet[d] = vcat(musPropPerDim...)
    end
    angsProps = vcat(angsPropsSet...)
    musProps = vcat(musPropsSet...)

    angs = vcat(angsInit, angsProps...)
    mus = vcat(musInit, musProps...)

    (angs, mus)
end

function setrotations!(::TypeI, cc::Cnsolt{T,D}, angs::AbstractArray{T}, mus) where {D,T}
    # Initialization
    P = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nMuswu = fld(P,2)
    nAngswu = ngivensangles(nMuswu)
    nAngsb = fld(P,4)

    nParamsInit = ngivensangles(P)
    nParamsProps = ord .* (2*nAngswu + nAngsb)

    # set Cnsolt.initMatrices
    angsInit = angs[1:nParamsInit]
    musInit = mus[1:P]
    cc.initMatrices[1] = rotations2mat(angsInit, musInit, P)

    # set Cnsolt.propMatrices
    dimAngsRanges = intervals(nParamsProps, nParamsInit)
    dimMusRanges = intervals(ord .* P, P)

    for d = 1:D
        subAngsDim = view(angs, dimAngsRanges[d])
        subMusDim = view(mus, dimMusRanges[d])
        subAngsRanges = intervals(repeat([nAngswu, nAngswu, nAngsb], ord[d]))
        subMusRanges = intervals(repeat([nMuswu, nMuswu], ord[d]))
        for k = 1:ord[d]
            apw = view(subAngsDim, subAngsRanges[3k-2])
            apu = view(subAngsDim, subAngsRanges[3k-1])
            apb = view(subAngsDim, subAngsRanges[3k])

            mpw = view(subMusDim, subMusRanges[2k-1])
            mpu = view(subMusDim, subMusRanges[2k])

            cc.propMatrices[d][2*k-1] = rotations2mat(apw, mpw, fld(P,2))
            cc.propMatrices[d][2*k]   = rotations2mat(apu, mpu, fld(P,2))
            cc.paramAngles[d][k]      = apb
        end
    end

    return cc
end

function getrotations(::TypeII, cc::Cnsolt{T,D}) where {D,T}
    P = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-1),2)
    nParamsPropPerDimsOrder = (fld((P-1)*(P-3),4), fld((P+1)*(P-1),4)) .+ fld(P,4)
    nParamsProps = fld.(ord,2) .* sum(nParamsPropPerDimsOrder)
    # nParams = vcat(nParamsInit, nParamsProps...)

    # Angles ang MUS
    angsInit, musInit = mat2rotations(cc.initMatrices[1])

    angsPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    for d = 1:D
        nStages = fld(ord[d],2)
        angsPropPerDim = Array{Vector{T}}(undef, nStages)
        musPropPerDim = Array{Vector{T}}(undef, nStages)
        for k = 1:nStages
            (apw1, mpw1) = mat2rotations(cc.propMatrices[d][4*k-3])
            (apu1, mpu1) = mat2rotations(cc.propMatrices[d][4*k-2])
            (apw2, mpw2) = mat2rotations(cc.propMatrices[d][4*k-1])
            (apu2, mpu2) = mat2rotations(cc.propMatrices[d][4*k])
            apb1 = cc.paramAngles[d][2*k-1]
            apb2 = cc.paramAngles[d][2*k]

            angsPropPerDim[k] = vcat(apw1,apu1,apb1,apw2,apu2,apb2)
            musPropPerDim[k] = vcat(mpw1,mpu1,mpw2,mpu2)
        end
        angsPropsSet[d] = vcat(angsPropPerDim...)
        musPropsSet[d] = vcat(musPropPerDim...)
    end
    angsProps = vcat(angsPropsSet...)
    musProps = vcat(musPropsSet...)

    angs = vcat(angsInit, angsProps...)
    mus = vcat(musInit, musProps...)

    (angs, mus)
end

function setrotations!(::TypeII, cc::Cnsolt{T,D}, angs::AbstractArray{T}, mus) where {D,T}
    # Initialization
    P = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nMuswu1 = fld(P,2)
    nMuswu2 = cld(P,2)
    nAngswu1 = ngivensangles(nMuswu1)
    nAngswu2 = ngivensangles(nMuswu2)
    nAngsb = fld(P,4)

    nParamsInit = ngivensangles(P)
    nParamsProps = fld.(ord,2) .* (2 * (nAngswu1 + nAngswu2 + nAngsb))

    # set Cnsolt.initMatrices
    angsInit = angs[1:nParamsInit]
    musInit = mus[1:P]
    cc.initMatrices[1] = rotations2mat(angsInit, musInit, P)

    # set Cnsolt.propMatrices
    dimAngsRanges = intervals(nParamsProps, nParamsInit)
    dimMusRanges = intervals(ord .* P, P)

    for d = 1:D
        nStages = fld(ord[d],2)
        subAngsDim = view(angs, dimAngsRanges[d])
        subMusDim = view(mus, dimMusRanges[d])
        subAngsRanges = intervals(repeat([nAngswu1, nAngswu1, nAngsb, nAngswu2, nAngswu2, nAngsb], nStages))
        subMusRanges = intervals(repeat([nMuswu1, nMuswu1, nMuswu2, nMuswu2], nStages))
        for k = 1:nStages
            apw1 = view(subAngsDim, subAngsRanges[6k-5])
            apu1 = view(subAngsDim, subAngsRanges[6k-4])
            apb1 = view(subAngsDim, subAngsRanges[6k-3])

            mpw1 = view(subMusDim, subMusRanges[4k-3])
            mpu1 = view(subMusDim, subMusRanges[4k-2])

            apw2 = view(subAngsDim, subAngsRanges[6k-2])
            apu2 = view(subAngsDim, subAngsRanges[6k-1])
            apb2 = view(subAngsDim, subAngsRanges[6k])

            mpw2 = view(subMusDim, subMusRanges[4k-1])
            mpu2 = view(subMusDim, subMusRanges[4k])

            cc.propMatrices[d][4*k-3] = rotations2mat(apw1, mpw1, fld(P,2))
            cc.propMatrices[d][4*k-2] = rotations2mat(apu1, mpu1, fld(P,2))
            cc.propMatrices[d][4*k-1] = rotations2mat(apw2, mpw2, cld(P,2))
            cc.propMatrices[d][4*k]   = rotations2mat(apu2, mpu2, cld(P,2))
            cc.paramAngles[d][2*k-1]  = apb1
            cc.paramAngles[d][2*k]    = apb2
        end
    end

    return cc
end

function getrotations(::TypeI, cc::Rnsolt{T,D}) where {D,T}
    P = sum(cc.nChannels)
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-2),4)
    nParamsPropPerDimsOrder = fld(P*(P-2),8)
    nParamsProps = ord .* nParamsPropPerDimsOrder
    # nParams = vcat(nParamsInit, nParamsProps...)

    # Angles ang MUS
    angsInitW, musInitW = mat2rotations(cc.initMatrices[1])
    angsInitU, musInitU = mat2rotations(cc.initMatrices[2])

    angsInit = vcat(angsInitW, angsInitU)
    musInit = vcat(musInitW, musInitU)

    angsPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    for d = 1:D
        angsPropPerDim = Array{Vector{T}}(undef, ord[d])
        musPropPerDim = Array{Vector{T}}(undef, ord[d])
        for k = 1:ord[d]
            (apu, mpu) = mat2rotations(cc.propMatrices[d][k])

            angsPropPerDim[k] = apu
            musPropPerDim[k] = mpu
        end
        angsPropsSet[d] = vcat(angsPropPerDim...)
        musPropsSet[d] = vcat(musPropPerDim...)
    end
    angsProps = vcat(angsPropsSet...)
    musProps = vcat(musPropsSet...)

    angs = vcat(angsInit, angsProps...)
    mus = vcat(musInit, musProps...)

    (angs, mus)
end

function setrotations!(::TypeI, cc::Rnsolt{T,D}, angs::AbstractArray{T}, mus) where {D,T}
    # Initialization
    nch = cc.nChannels
    P = sum(cc.nChannels)
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nAngsw, nAngsu = ngivensangles.(nch)
    nParamsInit = nAngsw + nAngsu
    nParamsProps = ord .* nAngsu

    # set Rnsolt.initMatrices
    initAngsRanges = intervals(ngivensangles.(nch))
    initMusRanges = intervals(nch)

    for idx = 1:2
        cc.initMatrices[idx] = rotations2mat(angs[initAngsRanges[idx]], mus[initMusRanges[idx]], nch[idx])
    end

    dimAngsRanges = intervals(nParamsProps, nParamsInit)
    dimMusRanges = intervals(ord .* fld(P,2), P)

    for d = 1:D
        subAngsDim = view(angs, dimAngsRanges[d])
        subMusDim = view(mus, dimMusRanges[d])
        subAngsRanges = intervals(fill(nAngsu, ord[d]))
        subMusRanges = intervals(fill(nch[2], ord[d]))
        for k = 1:ord[d]
            apu = view(subAngsDim, subAngsRanges[k])
            mpu = view(subMusDim, subMusRanges[k])

            cc.propMatrices[d][k] = rotations2mat(apu, mpu, nch[2])
        end
    end

    return cc
end

function getrotations(::TypeII, cc::Rnsolt{T,D}) where {D,T}
    nch = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = sum(fld.(nch .* (nch .- 1),2))
    nParamsPropPerDimsOrder = sum(fld.(nch .* (nch .- 1),2))
    nParamsProps = fld.(ord,2) .* sum(nParamsPropPerDimsOrder)
    # nParams = vcat(nParamsInit, nParamsProps...)

    # Angles ang MUS
    angsInitW, musInitW = mat2rotations(cc.initMatrices[1])
    angsInitU, musInitU = mat2rotations(cc.initMatrices[2])

    angsInit = vcat(angsInitW, angsInitU)
    musInit = vcat(musInitW, musInitU)

    angsPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    for d = 1:D
        nStages = fld(ord[d],2)
        angsPropPerDim = Array{Vector{T}}(undef, nStages)
        musPropPerDim = Array{Vector{T}}(undef, nStages)
        for k = 1:nStages
            (apu, mpu) = mat2rotations(cc.propMatrices[d][2*k-1])
            (apw, mpw) = mat2rotations(cc.propMatrices[d][2*k])

            angsPropPerDim[k] = vcat(apu,apw)
            musPropPerDim[k] = vcat(mpu,mpw)
        end
        angsPropsSet[d] = vcat(angsPropPerDim...)
        musPropsSet[d] = vcat(musPropPerDim...)
    end
    angsProps = vcat(angsPropsSet...)
    musProps = vcat(musPropsSet...)

    angs = vcat(angsInit, angsProps...)
    mus = vcat(musInit, musProps...)

    (angs, mus)
end

function setrotations!(::TypeII, cc::Rnsolt{T,D}, angs::AbstractArray{T}, mus) where {D,T}
    # Initialization
    nch = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    maxP, minP = if nch[1] > nch[2]
        (nch[1], nch[2])
    else
        (nch[2], nch[1])
    end

    nAngsu = ngivensangles(minP)
    nAngsw = ngivensangles(maxP)

    nParamsInit = ngivensangles.(nch)
    nParamsProps = fld.(ord,2) .* (nAngsu + nAngsw)

    initAngsRanges = intervals(nParamsInit)
    initMusRanges = intervals(nch)

    for idx = 1:2
        cc.initMatrices[idx] = rotations2mat(angs[initAngsRanges[idx]], mus[initMusRanges[idx]], nch[idx])
    end

    # set Cnsolt.propMatrices
    dimAngsRanges = intervals(nParamsProps, sum(nParamsInit))
    dimMusRanges = intervals(fld.(ord,2) .* sum(nch), sum(nch))

    for d = 1:D
        nStages = fld(ord[d],2)
        subAngsDim = angs[ dimAngsRanges[d] ]
        subMusDim = mus[ dimMusRanges[d] ]
        subAngsRanges = intervals(repeat([nAngsu, nAngsw], nStages))
        subMusRanges = intervals(repeat([minP, maxP], nStages))
        for k = 1:nStages
            apu = view(subAngsDim, subAngsRanges[2k-1])
            apw = view(subAngsDim, subAngsRanges[2k])

            mpu = view(subMusDim, subMusRanges[2k-1])
            mpw = view(subMusDim, subMusRanges[2k])

            cc.propMatrices[d][2k-1] = rotations2mat(apu, mpu, minP)
            cc.propMatrices[d][2k]   = rotations2mat(apw, mpw, maxP)
        end
    end

    return cc
end
