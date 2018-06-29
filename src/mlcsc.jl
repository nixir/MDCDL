#
# function mlista(mlcsc::MDCDL.MultiLayerCsc, y, gammaL0, K::Integer)
#     L = mlcsc.nLayers
#     dics = mlcsc.dictionaries
#
#     gamma = Array(L)
#     gamma[L] = gammaL0
#
#     hgamma = Array(L+1)
#     scs = Array(L)
#     for k = 1:K
#         hgamma[L+1] = gamma[L]
#         for l = L:-1:1
#             hgamma[l] = synthesize(dics[l], hgamma[l+1], scs[l])
#         end
#         gamma[0] = y
#
#         for i = 1:L
#             hoge = synthesize(dics[i], hgamma[i], scs[i]) - gamma[i-1]
#             fuga, sc = adjsynthesize(dics[i], hoge)
#             gamma[i] = mlcsc.proxOperator(hgamma[i] - fuga, lambda[i])
#         end
#     end
#     gamma[L]
# end
#
# function myunzip(x::Array{NTuple{N,T}}) where {N,T}
#     ntuple(N) do n
#         reshape([ x[l][n] for l in 1:length(x) ], size(x)...)
#     end
# end

function analyze(mlcsc::MDCDL.MultiLayerCsc{TC,D}, x::Array{TX,D}; isAllCoefs::Bool=false) where {TC,TX,D}
    gamma = Vector(mlcsc.nLayers+1)
    gamma[1] = x
    for l = 1:mlcsc.nLayers
        dic = mlcsc.dictionaries[l]

        gamma[l+1] = stepAnalysisBank(dic, gamma[l], outputMode="augumented")
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
        gamma[l] = stepSynthesisBank(dic, gamma[l+1], inputMode="augumented")
    end

    if isAllCoefs
        gamma
    else
        gamma[1]
    end
end
