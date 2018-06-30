function analyze(mlcsc::MDCDL.MultiLayerCsc{TC,D}, x::Array{TX,D}; isAllCoefs::Bool=false) where {TC,TX,D}
    gamma = Vector(mlcsc.nLayers+1)
    gamma[1] = x
    for l = 1:mlcsc.nLayers
        dic = mlcsc.dictionaries[l]

        gamma[l+1] = stepAnalysisBank(dic, gamma[l], outputMode=:augumented)
    end

    if isAllCoefs
        gamma
    else
        gamma[mlcsc.nLayers+1]
    end
end

function synthesize(mlcsc::MDCDL.MultiLayerCsc, y::Array; isAllCoefs::Bool=false)
    gamma = Vector(mlcsc.nLayers+1)
    gamma[mlcsc.nLayers+1] = y
    for l = mlcsc.nLayers:-1:1
        dic = mlcsc.dictionaries[l]
        gamma[l] = stepSynthesisBank(dic, gamma[l+1], inputMode=:augumented)
    end

    if isAllCoefs
        gamma
    else
        gamma[1]
    end
end
