using Plots, Images

function atmimshow(cc::MDCDL.Cnsolt{T,2,S}) where {S,T}
    P = cc.nChannels
    offset = 0.5

    afs = MDCDL.getAnalysisFilters(cc);

    plotsre = plot(map((f)->plot(Array{Gray{T}}(real.(f) .+ offset)), afs)...; layout=(1,P), aspect_ratio=:equal)
    plotsim = plot(map((f)->plot(Array{Gray{T}}(imag.(f) .+ offset)), afs)...; layout=(1,P), aspect_ratio=:equal)

    plot(plotsre, plotsim; layout=(2,1), aspect_ratio=:equal)
end

function atmimshow(cc::MDCDL.Rnsolt{T,2,S}) where {S,T}
    nch = cc.nChannels
    offset = 0.5

    afs = MDCDL.getAnalysisFilters(cc);

    plotatms = map((f)->plot(Array{Gray{T}}(f .+ offset)), afs)
    plotsyms = plot(plotatms[1:nch[1]]...; layout=(1,nch[1]))
    plotasyms = plot(plotatms[nch[1]+1:end]...; layout=(1,nch[2]))

    plot(plotsyms, plotasyms; layout=(2,1); )
end

# function atmimshow(mlcsc::MDCDL.MultiLayerCsc, args...)
#     for l = mlcsc.nLayers
#         atmimshow(mlcsc.dictionaries[l], args...)
#     end
# end
