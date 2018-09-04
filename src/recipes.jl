using RecipesBase
using ColorTypes

@recipe function atmimshow(cc::Cnsolt{T,2,S}) where {T,S}
    nch = cc.nChannels
    layout := (2,nch)
    clims := (-1.0, 1.0)

    afs = getAnalysisFilters(cc)
    afsre = map(f->real.(f), afs)
    afsim = map(f->imag.(f), afs)

    mafsre = map(f-> 0.5*f .+ 0.5, afsre)
    mafsim = map(f-> 0.5*f .+ 0.5, afsim)

    atms = Array{Gray{Float64}}[ mafsre; mafsim ]

    for idx = 1:2*nch
        @series begin
            subplot := idx
            aspect_ratio := :equal
            atms[idx]
        end
    end
end

@recipe function atmimshow(cc::Rnsolt{T,2,S}) where {T,S}
    mxP = maximum(cc.nChannels)
    layout := (2,mxP)
    clims := (-1.0, 1.0)
    nch = cc.nChannels
    difch = nch[2]-nch[1]

    afs = getAnalysisFilters(cc)

    mafs = map(f-> 0.5*f .+ 0.5, afs)

    dummyimg = fill(0.0, size(afs[1]))

    afssym = Array{Gray{Float64}}[ mafs[1:nch[1]]; fill(dummyimg, max(difch, 0)) ]
    afsasym = Array{Gray{Float64}}[ mafs[(nch[1]+1):end]; fill(dummyimg, max(-difch, 0)) ]

    for idx = 1:mxP
        @series begin
            subplot := idx
            aspect_ratio := :equal
            afssym[idx]
        end

        @series begin
            subplot := idx+mxP
            aspect_ratio := :equal
            afsasym[idx]
        end
    end
end
