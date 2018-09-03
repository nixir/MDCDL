using Plots: plot, px, heatmap
using ColorTypes

function atmimshow(cc::MDCDL.Cnsolt{T,2,S}; clims=(-1.0,1.0), coordinate=:cartesian) where {S,T}
    P = cc.nChannels

    afs = getAnalysisFilters(cc);
    atms = if coordinate == :cartesian
        [ real.(afs) ; imag.(afs) ]
    elseif coordinate == :polar
        rs = map(f->abs.(f) .- 0.5, afs)
        angs = map(f->angle.(f) ./ pi, afs)
        [ rs; angs ]
    end

    plot(heatmap.(atms, color=:gray, aspect_ratio=:equal, legend=false, clims=clims)...; layout=(2,P))
end

function atmimshow(cc::MDCDL.Rnsolt{T,2,S}; clims=(-1.0, 1.0)) where {S,T}
    nch = cc.nChannels
    difch = nch[2]-nch[1]

    afs = getAnalysisFilters(cc);

    dummyimg = fill(-Inf, size(afs[1]))

    afssym = [ afs[1:nch[1]]; fill(dummyimg, max(difch, 0)) ]
    afsasym = [ afs[(nch[1]+1):end]; fill(dummyimg, max(-difch, 0)) ]

    plotsyms = plot(heatmap.(afssym, color=:gray, aspect_ratio=:equal, legend=false, clims=clims)..., layout=(1,length(afssym)))
    plotasyms = plot(heatmap.(afsasym, color=:gray, aspect_ratio=:equal, legend=false, clims=clims)..., layout=(1,length(afsasym)))

    plot(plotsyms, plotasyms; layout=(2,1), aspect_ratio=:equal, margin=0px)
end
