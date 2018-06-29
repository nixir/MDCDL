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

# function analyzeNew(mlcsc::MDCDL.MultiLayerCsc{TC,D}, x::Array{TX,D}; isAllCoefs::Bool=false) where {TC,TX,D}
#     szData = size(x)
#     gamma = Vector(mlcsc.nLayers+1)
#     gamma[1] = x
#     for l = 1:mlcsc.nLayers
#         dic = mlcsc.dictionaries[l]
#         df = dic.decimationFactor
#         M = prod(dic.decimationFactor)
#         nBlocks = fld.(szData,df)
#
#         blkx = MDCDL.array2vecblocks(gamma[l],df)
#         tx = reshape(blkx, M, cld(length(gamma[l]),M))
#         ty = MDCDL.multipleAnalysisPolyphaseMat(dic, tx, nBlocks)
#
#         szData = tuple(nBlocks..., sum(dic.nChannels))
#
#         gamma[l+1] = MDCDL.vecblocks2array(ty.',szData,tuple(ones(Integer, D+l)...))
#     end
#
#     if isAllCoefs
#         error("isAllCoefs = $isAllCoefs: this option is not implemented.")
#     else
#         gamma[mlcsc.nLayers+1]
#     end
# end

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
