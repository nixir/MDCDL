using Plots: plot, px
using ColorTypes

function atmimshow(cc::MDCDL.Cnsolt{T,2,S}) where {S,T}
    P = cc.nChannels
    offset = 0.5

    afs = getAnalysisFilters(cc);

    atmsre = [ Array{Gray{T}}(real.(f) .+ offset) for f in afs ]
    atmsim = [ Array{Gray{T}}(imag.(f) .+ offset) for f in afs ]

    plotsre = plot(plot.(atmsre; ticks=nothing)...; layout=(1,P), aspect_ratio=:equal)
    plotsim = plot(plot.(atmsim; ticks=nothing)...; layout=(1,P), aspect_ratio=:equal)

    plot(plotsre, plotsim; layout=(2,1), aspect_ratio=:equal)
end

function atmimshow(cc::MDCDL.Rnsolt{T,2,S}) where {S,T}
    nch = cc.nChannels
    offset = 0.5

    afs = getAnalysisFilters(cc);

    atms = [ Array{Gray{T}}(f .+ offset) for f in afs ]

    plotsyms = plot(plot.(atms[1:nch[1]]; ticks=nothing, margin=0px)...; layout=(1,nch[1]), aspect_ratio=:equal, margin=0px)
    plotasyms = plot(plot.(atms[nch[1]+1:end]; ticks=nothing, margin=0px)...; layout=(1,nch[2]), aspect_ratio=:equal, margin=0px)

    plot(plotsyms, plotasyms; layout=(2,1), aspect_ratio=:equal, margin=0px)
end

# function atmimshow(mlcsc::MDCDL.MultiLayerCsc, args...)
#     for l = mlcsc.nLayers
#         atmimshow(mlcsc.dictionaries[l], args...)
#     end
# end
