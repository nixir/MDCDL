using RecipesBase
using ColorTypes, ColorSchemes
using TiledIteration

@recipe function atmimshow(cc::Cnsolt{T,2,S}; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10) where {T,S}
    nch = cc.nChannels
    ord = cc.polyphaseOrder
    df =  cc.decimationFactor

    layout :=  (2,nch)
    size -->  20 .* df .* (ord .+ 1) .* (nch, 2)

    afs = getAnalysisFilters(cc)
    afsre = map(f->real.(f), afs)
    afsim = map(f->imag.(f), afs)

    mafsre = map(f->get(cscheme, f, rangescale), afsre)
    mafsim = map(f->get(cscheme, f, rangescale), afsim)

    atms = [ mafsre; mafsim ]
    atmsup = resize_by_nn.(atms, atomscale)

    for idx = 1:2*nch
        @series begin
            subplot := idx
            axis    := false
            grid    := false

            aspect_ratio := :equal
            atmsup[idx]
        end
    end
end

@recipe function atmimshow(cc::Cnsolt{T,2,S}, p::Integer; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10) where {T,S}
    nch = cc.nChannels
    ord = cc.polyphaseOrder
    df =  cc.decimationFactor

    layout :=  (2,1)
    size -->  20 .* df .* (ord .+ 1) .* (1,2)

    afs = getAnalysisFilters(cc)
    atm = afs[p]

    mafsre = get(cscheme, real(atm), rangescale)
    mafsim = get(cscheme, imag(atm), rangescale)

    atms = [ mafsre, mafsim ]
    atmsup = resize_by_nn.(atms, atomscale)

    for idx = 1:2
        @series begin
            subplot := idx
            axis    := false
            grid    := false

            aspect_ratio := :equal
            atmsup[idx]
        end
    end
end

@recipe function atmimshow(cc::Rnsolt{T,2,S}; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10) where {T,S}
    mxP = maximum(cc.nChannels)
    ord = cc.polyphaseOrder
    df =  cc.decimationFactor

    layout     :=  (2,mxP)
    size -->  20 .* df .* (ord .+ 1) .* (mxP, 2)

    nch = cc.nChannels
    difch = nch[2]-nch[1]

    afs = getAnalysisFilters(cc)

    dummyimg = fill(-Inf, size(afs[1]))
    afssym = [ afs[1:nch[1]]; fill(dummyimg, max(difch, 0)) ]
    afsasym = [ afs[(nch[1]+1):end]; fill(dummyimg, max(-difch, 0)) ]

    mafssym = map(f->get(cscheme, f, rangescale), afssym)
    mafsasym = map(f->get(cscheme, f, rangescale), afsasym)

    mafssymup = resize_by_nn.(mafssym, atomscale)
    mafsasymup = resize_by_nn.(mafsasym, atomscale)

    for idx = 1:mxP
        @series begin
            subplot := idx
            axis    := false
            grid    := false
            aspect_ratio := :equal

            mafssym[idx]
        end

        @series begin
            subplot := idx+mxP
            axis    := false
            grid    := false
            aspect_ratio := :equal

            mafsasym[idx]
        end
    end
end

@recipe function atmimshow(cc::Rnsolt{T,2,S}, p::Integer; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10) where {T,S}
    ord = cc.polyphaseOrder
    df =  cc.decimationFactor

    axis    := false
    grid    := false
    
    aspect_ratio := :equal
    size -->  20 .* df .* (ord .+ 1)

    afs = getAnalysisFilters(cc)
    atm = afs[p]

    mafssym = get(cscheme, atm, rangescale)
    mafssymup = resize_by_nn(mafssym, atomscale)

    mafssymup
end

function resize_by_nn(x::AbstractArray{T,D}, scale::Integer) where {T,D}
    output = similar(x, T, size(x) .* scale)
    tiles = collect(TileIterator(axes(output), (fill(scale,D)...,)))

    for idx in 1:length(tiles)
        output[tiles[idx]...] .= x[idx]
    end
    output
end
