
#TODO: 角度パラメータのベクトル化手法の仕様をどこかに記述する．
#TODO: コードが汚いのでリファクタリングする
function getAngleParameters(cc::MDCDL.Cnsolt{T,D,:TypeI}) where {D,T}
    P = cc.nChannels
    df = cc.decimationFactor
    ord = cc.polyphaseOrder

    nParamsInit = fld(P*(P-1),2)
    nParamsPropPerDimsOrder = fld(P*(P-2),4) + fld(P,4)
    nParamsProps = ord .* nParamsPropPerDimsOrder
    # nParams = vcat(nParamsInit, nParamsProps...)

    # Angles ang MUS
    angsInit, musInit = MDCDL.mat2rotations(cc.initMatrices[1])

    angsPropsSet = [ zeros(npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(npk) for npk in nParamsProps ]
    for d = 1:D
        angsPropPerDim = Array{Vector}(ord[d])
        musPropPerDim = Array{Vector}(ord[d])
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
function setAngleParameters!(cc::MDCDL.Cnsolt{T,D,:TypeI}, angs::Vector{T}, mus) where {D,T}
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
    dimAngsRanges = [ colon(delimitersAngs[d]+1, delimitersAngs[d+1]) + nParamsInit for d in 1:D]

    delimitersMus = cumsum([ 0, (ord .* P)... ])
    dimMusRanges = [ colon(delimitersMus[d]+1, delimitersMus[d+1]) + P for d in 1:D]

    nAngswu = fld(P*(P-2),8)
    nAngsb = fld(P,4)
    nMuswu = fld(P,2)
    for d = 1:D
        subAngsDim = angs[ dimAngsRanges[d] ]
        subMusDim = mus[ dimMusRanges[d] ]
        for k = 1:ord[d]
            subAngsOrd = subAngsDim[(1:nParamsPropPerDimsOrder) + (k-1)*nParamsPropPerDimsOrder]
            subMusOrd = subMusDim[(1:P) + (k-1)*P];

            apw = subAngsOrd[1:nAngswu]
            apu = subAngsOrd[nAngswu+1:2*nAngswu]
            apb = subAngsOrd[2*nAngswu+1:nParamsPropPerDimsOrder]

            mpw = subMusOrd[1:nMuswu];
            mpu = subMusOrd[nMuswu+1:2*nMuswu];

            cc.propMatrices[d][2*k-1] = MDCDL.rotations2mat(apw, mpw)
            cc.propMatrices[d][2*k]   = MDCDL.rotations2mat(apu, mpu)
            cc.paramAngles[d][k]      = apb
        end
    end

    return cc
end

function getAngleParameters(cc::MDCDL.Rnsolt{T,D,:TypeI}) where {D,T}
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

    angsPropsSet = [ zeros(npk) for npk in nParamsProps ]
    musPropsSet = [ zeros(npk) for npk in nParamsProps ]
    for d = 1:D
        angsPropPerDim = Array{Vector}(ord[d])
        musPropPerDim = Array{Vector}(ord[d])
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
function setAngleParameters!(cc::MDCDL.Rnsolt{T,D,:TypeI}, angs::Vector{T}, mus) where {D,T}
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
    dimAngsRanges = [ colon(delimitersAngs[d]+1, delimitersAngs[d+1]) + nParamsInit for d in 1:D]

    delimitersMus = cumsum([ 0, (ord .* fld(P,2))... ])
    dimMusRanges = [ colon(delimitersMus[d]+1, delimitersMus[d+1]) + fld(P,2) for d in 1:D]


    nAngsu = fld(P*(P-2),8)
    nMusu = fld(P,2)
    for d = 1:D
        subAngsDim = angs[ dimAngsRanges[d] ]
        subMusDim = mus[ dimMusRanges[d] ]
        for k = 1:ord[d]
            subAngsOrd = subAngsDim[(1:nParamsPropPerDimsOrder) + (k-1)*nParamsPropPerDimsOrder]
            subMusOrd = subMusDim[(1:fld(P,2)) + (k-1)*fld(P,2)]

            apu = subAngsOrd[1:nAngsu]
            mpu = subMusOrd[1:nMusu]

            cc.propMatrices[d][k] = MDCDL.rotations2mat(apu, mpu)
        end
    end

    return cc
end
