using RecipesBase
using ColorTypes, ColorSchemes
using TiledIteration

@recipe function atmimshow(cc::Cnsolt{T,2}; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10, coordinate=:cartesian) where {T}
    nch = cc.nChannels
    ord = cc.polyphaseOrder
    df =  cc.decimationFactor

    layout :=  (2,nch)
    size -->  20 .* df .* (ord .+ 1) .* (nch, 2)

    afs = analysiskernel(cc)

    mafsup, mafslw = if coordinate == :cartesian
        afsup = map(f->real.(f), afs)
        afslw = map(f->imag.(f), afs)
        mafsup = map(f->get(cscheme, f, rangescale), afsup)
        mafslw = map(f->get(cscheme, f, rangescale), afslw)
        (mafsup, mafslw)
    elseif coordinate == :polar
        mxv = norm(rangescale)
        afsr = map(f->abs.(f), afs)
        afsa = map(f->angle.(f), afs)
        mafsup = map(f->get(cscheme, f, (0, mxv)), afsr)
        mafslw = map((fr, fa)->RGB.(HSV.(180 .* (fa ./ pi .+ 1), 1.0, fr ./ mxv)), afsr, afsa)

        (mafsup, mafslw)
    end

    atms = [ mafsup; mafslw ]
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

@recipe function atmimshow(cc::Cnsolt{T,2}, p::Integer; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10, coordinate=:cartesian) where {T}
    nch = cc.nChannels
    ord = cc.polyphaseOrder
    df =  cc.decimationFactor

    layout :=  (2,1)
    size -->  20 .* df .* (ord .+ 1) .* (1,2)

    afs = analysiskernel(cc)
    atm = afs[p]

    mafsup, mafslw = if coordinate == :cartesian
        mafsup = get(cscheme, real(atm), rangescale)
        mafslw = get(cscheme, imag(atm), rangescale)
        (mafsup, mafslw)
    elseif coordinate == :polar
        mxv = norm(rangescale)
        afsr = abs.(atm)
        afsa = angle.(atm)
        mafsup = get(cscheme, afsr, (0, mxv))
        mafslw = RGB.(HSV.(180 .* (afsa ./ pi .+ 1), 1.0, afsr ./ mxv))

        (mafsup, mafslw)
    end


    atms = [ mafsup, mafslw ]
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

@recipe function atmimshow(cc::Rnsolt{T,2}; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10) where {T}
    mxP = maximum(cc.nChannels)
    ord = cc.polyphaseOrder
    df =  cc.decimationFactor

    layout     :=  (2,mxP)
    size -->  20 .* df .* (ord .+ 1) .* (mxP, 2)

    nch = cc.nChannels
    difch = nch[2]-nch[1]

    afs = analysiskernel(cc)

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

@recipe function atmimshow(cc::Rnsolt{T,2}, p::Integer; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10) where {T}
    ord = cc.polyphaseOrder
    df =  cc.decimationFactor

    axis    := false
    grid    := false

    aspect_ratio := :equal
    size -->  20 .* df .* (ord .+ 1)

    afs = analysiskernel(cc)
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
