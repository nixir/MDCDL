
#TODO: 角度パラメータのベクトル化手法の仕様をどこかに記述する．
#TODO: コードが汚いのでリファクタリングする

getAngleParameters(cc::AbstractNsolt) = getAngleParameters(Val{cc.category}, cc)
setAngleParameters!(cc::AbstractNsolt, θ, μ) = setAngleParameters!(Val{cc.category}, cc, θ, μ)

setAngleParameters(cc::AbstractNsolt, args...) = setAngleParameters!(deepcopy(cc), args...)

function getAngleParameters(::Type{Val{:TypeI}}, cc::MDCDL.Cnsolt{T,D}) where {D,T}
    P = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-1),2)
    nParamsPropPerDimsOrder = fld(P*(P-2),4) + fld(P,4)
    nParamsProps = ord .* nParamsPropPerDimsOrder
    # nParams = vcat(nParamsInit, nParamsProps...)

    # Angles ang MUS
    angsInit, musInit = MDCDL.mat2rotations(cc.initMatrices[1])

    angsPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    for d = 1:D
        angsPropPerDim = Array{Vector{T}}(undef, ord[d])
        musPropPerDim = Array{Vector{T}}(undef, ord[d])
        for k = 1:ord[d]
            (apw, mpw) = MDCDL.mat2rotations(cc.propMatrices[d][2*k-1])
            (apu, mpu) = MDCDL.mat2rotations(cc.propMatrices[d][2*k])
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

#TODO: コードが汚いのでリファクタリングする
function setAngleParameters!(::Type{Val{:TypeI}}, cc::MDCDL.Cnsolt{T,D}, angs::AbstractArray{T}, mus) where {D,T}
    # Initialization
    P = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-1),2)
    nParamsPropPerDimsOrder = fld(P*(P-2),4) + fld(P,4)
    nParamsProps = ord .* nParamsPropPerDimsOrder
    nParams = vcat(nParamsInit, nParamsProps...)

    # set Cnsolt.initMatrices
    angsInit = angs[1:nParamsInit]
    musInit = mus[1:P]
    cc.initMatrices[1] = MDCDL.rotations2mat(angsInit, musInit)

    # set Cnsolt.propMatrices
    delimitersAngs = cumsum([ 0, nParamsProps... ])
    dimAngsRanges = [ ((delimitersAngs[d]+1):delimitersAngs[d+1]) .+ nParamsInit for d in 1:D]

    delimitersMus = cumsum([ 0, (ord .* P)... ])
    dimMusRanges = [ ((delimitersMus[d]+1):delimitersMus[d+1]) .+ P for d in 1:D]

    nAngswu = fld(P*(P-2),8)
    nAngsb = fld(P,4)
    nMuswu = fld(P,2)
    for d = 1:D
        subAngsDim = angs[ dimAngsRanges[d] ]
        subMusDim = mus[ dimMusRanges[d] ]
        for k = 1:ord[d]
            subAngsOrd = subAngsDim[(1:nParamsPropPerDimsOrder) .+ (k-1)*nParamsPropPerDimsOrder]
            subMusOrd = subMusDim[(1:P) .+ (k-1)*P]

            apw = subAngsOrd[1:nAngswu]
            apu = subAngsOrd[(nAngswu+1):2*nAngswu]
            apb = subAngsOrd[(2*nAngswu+1):nParamsPropPerDimsOrder]

            mpw = subMusOrd[1:nMuswu]
            mpu = subMusOrd[(nMuswu+1):2*nMuswu]

            cc.propMatrices[d][2*k-1] = MDCDL.rotations2mat(apw, mpw)
            cc.propMatrices[d][2*k]   = MDCDL.rotations2mat(apu, mpu)
            cc.paramAngles[d][k]      = apb
        end
    end

    return cc
end

function getAngleParameters(::Type{Val{:TypeII}}, cc::MDCDL.Cnsolt{T,D}) where {D,T}
    P = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-1),2)
    nParamsPropPerDimsOrder = (fld((P-1)*(P-3),4), fld((P+1)*(P-1),4)) .+ fld(P,4)
    nParamsProps = fld.(ord,2) .* sum(nParamsPropPerDimsOrder)
    # nParams = vcat(nParamsInit, nParamsProps...)

    # Angles ang MUS
    angsInit, musInit = MDCDL.mat2rotations(cc.initMatrices[1])

    angsPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    for d = 1:D
        nStages = fld(ord[d],2)
        angsPropPerDim = Array{Vector{T}}(undef, nStages)
        musPropPerDim = Array{Vector{T}}(undef, nStages)
        for k = 1:nStages
            (apw1, mpw1) = MDCDL.mat2rotations(cc.propMatrices[d][4*k-3])
            (apu1, mpu1) = MDCDL.mat2rotations(cc.propMatrices[d][4*k-2])
            (apw2, mpw2) = MDCDL.mat2rotations(cc.propMatrices[d][4*k-1])
            (apu2, mpu2) = MDCDL.mat2rotations(cc.propMatrices[d][4*k])
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

function setAngleParameters!(::Type{Val{:TypeII}}, cc::Cnsolt{T,D}, angs::AbstractArray{T}, mus) where {D,T}
    # Initialization
    P = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-1),2)
    nParamsPropPerDimsOrder = (fld((P-1)*(P-3),4), fld((P+1)*(P-1),4)) .+ fld(P,4)
    nParamsProps = fld.(ord,2) .* sum(nParamsPropPerDimsOrder)
    nParams = vcat(nParamsInit, nParamsProps...)

    # set Cnsolt.initMatrices
    angsInit = angs[1:nParamsInit]
    musInit = mus[1:P]
    cc.initMatrices[1] = MDCDL.rotations2mat(angsInit, musInit)

    # set Cnsolt.propMatrices
    delimitersAngs = cumsum([ 0, nParamsProps... ])
    dimAngsRanges = [ ((delimitersAngs[d]+1):delimitersAngs[d+1]) .+ nParamsInit for d in 1:D]

    delimitersMus = cumsum([ 0, (ord .* P)... ])
    dimMusRanges = [ ((delimitersMus[d]+1):delimitersMus[d+1]) .+ P for d in 1:D]

    nAngswu1 = fld((P-1)*(P-3),8)
    nAngswu2 = fld((P+1)*(P-1),8)
    nAngsb = fld(P,4)
    nMuswu1 = fld(P,2)
    nMuswu2 = cld(P,2)
    for d = 1:D
        nStages = fld(ord[d],2)
        subAngsDim = angs[ dimAngsRanges[d] ]
        subMusDim = mus[ dimMusRanges[d] ]
        for k = 1:nStages
            subAngsOrd1 = subAngsDim[(1:nParamsPropPerDimsOrder[1]) .+ (k-1)*sum(nParamsPropPerDimsOrder)]
            subMusOrd1 = subMusDim[(1:P-1) .+ (k-1)*2*P]
            subAngsOrd2 = subAngsDim[(1:nParamsPropPerDimsOrder[2]) .+ ((k-1)*sum(nParamsPropPerDimsOrder) + nParamsPropPerDimsOrder[1])]
            subMusOrd2 = subMusDim[(1:P+1) .+ ((P-1) + (k-1)*2*P)]

            apw1 = subAngsOrd1[1:nAngswu1]
            apu1 = subAngsOrd1[nAngswu1+1:2*nAngswu1]
            apb1 = subAngsOrd1[2*nAngswu1+1:2*nAngswu1+nAngsb]

            mpw1 = subMusOrd1[1:nMuswu1]
            mpu1 = subMusOrd1[nMuswu1+1:2*nMuswu1]

            apw2 = subAngsOrd2[1:nAngswu2]
            apu2 = subAngsOrd2[nAngswu2+1:2*nAngswu2]
            apb2 = subAngsOrd2[2*nAngswu2+1:2*nAngswu2+nAngsb]

            mpw2 = subMusOrd2[1:nMuswu2]
            mpu2 = subMusOrd2[nMuswu2+1:2*nMuswu2]

            cc.propMatrices[d][4*k-3] = MDCDL.rotations2mat(apw1, mpw1)
            cc.propMatrices[d][4*k-2] = MDCDL.rotations2mat(apu1, mpu1)
            cc.propMatrices[d][4*k-1] = MDCDL.rotations2mat(apw2, mpw2)
            cc.propMatrices[d][4*k]   = MDCDL.rotations2mat(apu2, mpu2)
            cc.paramAngles[d][2*k-1]  = apb1
            cc.paramAngles[d][2*k]    = apb2
        end
    end

    return cc
end

function getAngleParameters(::Type{Val{:TypeI}}, cc::MDCDL.Rnsolt{T,D}) where {D,T}
    P = sum(cc.nChannels)
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-2),4)
    nParamsPropPerDimsOrder = fld(P*(P-2),8)
    nParamsProps = ord .* nParamsPropPerDimsOrder
    # nParams = vcat(nParamsInit, nParamsProps...)

    # Angles ang MUS
    angsInitW, musInitW = MDCDL.mat2rotations(cc.initMatrices[1])
    angsInitU, musInitU = MDCDL.mat2rotations(cc.initMatrices[2])

    angsInit = vcat(angsInitW, angsInitU)
    musInit = vcat(musInitW, musInitU)

    angsPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    for d = 1:D
        angsPropPerDim = Array{Vector{T}}(undef, ord[d])
        musPropPerDim = Array{Vector{T}}(undef, ord[d])
        for k = 1:ord[d]
            (apu, mpu) = MDCDL.mat2rotations(cc.propMatrices[d][k])

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

#TODO: コードが汚いのでリファクタリングする
function setAngleParameters!(::Type{Val{:TypeI}}, cc::MDCDL.Rnsolt{T,D}, angs::AbstractArray{T}, mus) where {D,T}
    # Initialization
    P = sum(cc.nChannels)
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-2),4)
    nParamsPropPerDimsOrder = fld(P*(P-2),8)
    nParamsProps = ord .* nParamsPropPerDimsOrder
    # nParams = vcat(nParamsInit, nParamsProps...)

    # set Cnsolt.initMatrices
    angsInitW = angs[1:fld(nParamsInit,2)]
    musInitW = mus[1:fld(P,2)]
    angsInitU = angs[fld(nParamsInit,2)+1:nParamsInit]
    musInitU = mus[fld(P,2)+1:P]
    cc.initMatrices[1] = MDCDL.rotations2mat(angsInitW, musInitW)
    cc.initMatrices[2] = MDCDL.rotations2mat(angsInitU, musInitU)

    # set Cnsolt.propMatrices
    delimitersAngs = cumsum([ 0, nParamsProps... ])
    dimAngsRanges = [ ((delimitersAngs[d]+1):delimitersAngs[d+1]) .+ nParamsInit for d in 1:D]

    delimitersMus = cumsum([ 0, (ord .* fld(P,2))... ])
    dimMusRanges = [ ((delimitersMus[d]+1):delimitersMus[d+1]) .+ P for d in 1:D]


    nAngsu = fld(P*(P-2),8)
    nMusu = fld(P,2)
    for d = 1:D
        subAngsDim = angs[ dimAngsRanges[d] ]
        subMusDim = mus[ dimMusRanges[d] ]
        for k = 1:ord[d]
            subAngsOrd = subAngsDim[(1:nParamsPropPerDimsOrder) .+ (k-1)*nParamsPropPerDimsOrder]
            subMusOrd = subMusDim[(1:fld(P,2)) .+ (k-1)*fld(P,2)]

            apu = subAngsOrd[1:nAngsu]
            mpu = subMusOrd[1:nMusu]

            cc.propMatrices[d][k] = MDCDL.rotations2mat(apu, mpu)
        end
    end

    return cc
end

function getAngleParameters(::Type{Val{:TypeII}}, cc::MDCDL.Rnsolt{T,D}) where {D,T}
    nch = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = sum(fld.(nch .* (nch .- 1),2))
    nParamsPropPerDimsOrder = sum(fld.(nch .* (nch .- 1),2))
    nParamsProps = fld.(ord,2) .* sum(nParamsPropPerDimsOrder)
    # nParams = vcat(nParamsInit, nParamsProps...)

    # Angles ang MUS
    angsInitW, musInitW = MDCDL.mat2rotations(cc.initMatrices[1])
    angsInitU, musInitU = MDCDL.mat2rotations(cc.initMatrices[2])

    angsInit = vcat(angsInitW, angsInitU)
    musInit = vcat(musInitW, musInitU)

    angsPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(T, npk) for npk in nParamsProps ]
    for d = 1:D
        nStages = fld(ord[d],2)
        angsPropPerDim = Array{Vector{T}}(undef, nStages)
        musPropPerDim = Array{Vector{T}}(undef, nStages)
        for k = 1:nStages
            (apu, mpu) = MDCDL.mat2rotations(cc.propMatrices[d][2*k-1])
            (apw, mpw) = MDCDL.mat2rotations(cc.propMatrices[d][2*k])

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

function setAngleParameters!(::Type{Val{:TypeII}}, cc::MDCDL.Rnsolt{T,D}, angs::AbstractArray{T}, mus) where {D,T}
    # Initialization
    nch = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    maxP, minP = if nch[1] > nch[2]
        (nch[1], nch[2])
    else
        (nch[2], nch[1])
    end

    nParamsInit = fld.(nch .* (nch .- 1),2)
    nParamsPropPerDimsOrder = sum(fld.(nch .* (nch .- 1),2))
    nParamsProps = fld.(ord,2) .* sum(nParamsPropPerDimsOrder)
    nParams = vcat(sum(nParamsInit), nParamsProps...)

    # set Cnsolt.initMatrices
    angsInitW = angs[1:nParamsInit[1]]
    musInitW = mus[1:nch[1]]
    angsInitU = angs[(1:nParamsInit[2]) .+ nParamsInit[1]]
    musInitU = mus[(1:nch[2]) .+ nch[1]]

    cc.initMatrices[1] = MDCDL.rotations2mat(angsInitW, musInitW)
    cc.initMatrices[2] = MDCDL.rotations2mat(angsInitU, musInitU)

    # set Cnsolt.propMatrices
    delimitersAngs = cumsum([ 0, nParamsProps... ])
    dimAngsRanges = [ ((delimitersAngs[d]+1):delimitersAngs[d+1]) .+ sum(nParamsInit) for d in 1:D]

    delimitersMus = cumsum([ 0, (fld.(ord,2) .* sum(nch))... ])
    dimMusRanges = [ ((delimitersMus[d]+1):delimitersMus[d+1]) .+ sum(nch) for d in 1:D]
    nAngsu = fld(minP*(minP-1),2)
    nAngsw = fld(maxP*(maxP-1),2)

    for d = 1:D
        nStages = fld(ord[d],2)
        subAngsDim = angs[ dimAngsRanges[d] ]
        subMusDim = mus[ dimMusRanges[d] ]
        for k = 1:nStages
            subAngsOrd1 = subAngsDim[(1:nAngsu) .+ (k-1)*sum(nParamsPropPerDimsOrder)]
            subMusOrd1 = subMusDim[(1:minP) .+ (k-1)*sum(nch)]
            subAngsOrd2 = subAngsDim[(1:nAngsw) .+ ((k-1)*sum(nParamsPropPerDimsOrder) + nAngsu)]
            subMusOrd2 = subMusDim[(1:maxP) .+ (minP + (k-1)*sum(nch))]

            apu = subAngsOrd1[1:nAngsu]
            apw = subAngsOrd2[1:nAngsw]

            mpu = subMusOrd1[1:minP]
            mpw = subMusOrd2[1:maxP]

            cc.propMatrices[d][2*k-1] = MDCDL.rotations2mat(apu, mpu)
            cc.propMatrices[d][2*k]   = MDCDL.rotations2mat(apw, mpw)
        end
    end

    return cc
end
