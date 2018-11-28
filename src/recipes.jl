using RecipesBase
using ColorTypes, ColorSchemes
using TiledIteration

@recipe function atmimshow(cc::Cnsolt{T,2}; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10, coordinate=:cartesian) where {T}
    nch = nchannels(cc)
    ord = orders(cc)
    df =  decimations(cc)

    layout :=  (2,nch)
    size -->  100 .* df .* (ord .+ 1) .* (nch, 2)

    atmsreim = map(analysiskernels(cc)) do fp
        apply_colorscheme(Val{coordinate}, fp, cscheme, rangescale, atomscale)
    end

    atmsup = [ [ f[1] for f in atmsreim ]..., [ f[2] for f in atmsreim ]... ]

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
    nch = nchannels(cc)
    ord = orders(cc)
    df =  decimations(cc)

    layout :=  (2,1)
    size -->  100 .* df .* (ord .+ 1) .* (1,2)

    afs = analysiskernels(cc)

    atmsup = apply_colorscheme(Val{coordinate}, afs[p], cscheme, rangescale, atomscale)

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

# @recipe function atmimshow(cc::Cnsolt{T,2}, ::Val{:onecolumn}; kwargs...)
#     atmimshow(cc; kwargs...)
# end

@recipe function atmimshow(cc::Rnsolt{T,2}; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10) where {T}
    mxP = maximum(cc.nChannels)
    nch = cc.nChannels
    ord = orders(cc)
    df =  decimations(cc)
    difch = nch[2]-nch[1]

    layout     :=  (2,mxP)
    size -->  100 .* df .* (ord .+ 1) .* (mxP, 2)

    afs = analysiskernels(cc)
    dummyimg = fill(-Inf, size(afs[1]))
    afssym = [ afs[1:nch[1]]; fill(dummyimg, max(difch, 0)) ]
    afsasym = [ afs[(nch[1]+1):end]; fill(dummyimg, max(-difch, 0)) ]

    afsup, afslw = map((afssym, afsasym,)) do fs
        map(f->apply_colorscheme(f, cscheme, rangescale, atomscale), fs)
    end

    for idx = 1:mxP
        @series begin
            subplot := idx
            axis    := false
            grid    := false
            aspect_ratio := :equal

            afsup[idx]
        end

        @series begin
            subplot := idx+mxP
            axis    := false
            grid    := false
            aspect_ratio := :equal

            afslw[idx]
        end
    end
end

@recipe function atmimshow(cc::Rnsolt{T,2}, p::Integer; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10) where {T}
    ord = orders(cc)
    df =  decimations(cc)

    axis    := false
    grid    := false

    aspect_ratio := :equal
    size -->  100 .* df .* (ord .+ 1)

    afs = analysiskernels(cc)

    apply_colorscheme(afs[p], cscheme, rangescale, atomscale)
end

@recipe function atmimshow(cc::Rnsolt{T,2}, ::Val{:onecolumn}; cscheme=ColorSchemes.gray, rangescale=(-0.5,0.5), atomscale=10) where {T}
    mxP = maximum(cc.nChannels)
    ord = orders(cc)
    df =  decimations(cc)
    nch = nchannels(cc)

    layout     :=  (1, nch)
    size -->  200 .* df .* (ord .+ 1) .* (nch, 1)

    afsout = map(f->apply_colorscheme(f, cscheme, rangescale, atomscale), analysiskernels(cc))

    for idx = 1:nch
        @series begin
            subplot := idx
            axis    := false
            grid    := false
            aspect_ratio := :equal

            afsout[idx]
        end
    end
end

apply_colorscheme(x::AbstractArray, args...; kwargs...) = apply_colorscheme(Val{:cartesian}, x, args...; kwargs...)
apply_colorscheme(::Type{Val{:cartesian}}, x::AbstractArray{T}, cscheme::AbstractVector{C}, rangescale::Tuple{R,R}, atomscale::Integer) where {T<:Complex,C<:Colorant,R<:Real} = map(t->apply_colorscheme(t, cscheme, rangescale, atomscale), reim(x))
function apply_colorscheme(::Type{Val{:cartesian}}, x::AbstractArray{T}, cscheme::AbstractVector{C}, rangescale::Tuple{R,R}, atomscale::Integer) where {T<:Real,C<:Colorant,R<:Real}
    resize_by_nn(get(cscheme, x, rangescale), atomscale)
end

function apply_colorscheme(::Type{Val{:polar}}, x::AbstractArray{T}, cscheme::AbstractVector{C}, rangescale::Tuple{R,R}, atomscale::Integer) where {T,C<:Colorant,R<:Real}
    mxv = norm(rangescale)
    rx = get(cscheme, abs.(x), (0, mxv))
    ax = RGB.(HSV.(x .|> angle .|> rad2deg .|> wrapdeg, 1.0, abs.(x) / mxv))
    resize_by_nn.((rx, ax), atomscale)
end

function resize_by_nn(x::AbstractArray{T,D}, scale::Integer) where {T,D}
    output = similar(x, T, size(x) .* scale)

    for (idx, tile) in enumerate(TileIterator(axes(output), (fill(scale,D)...,)))
        output[tile...] .= x[idx]
    end
    output
end

# wrapdeg(x::Real) = ifelse(x < 0, x+360, x)
wrapdeg(x::Real) = mod(x, 360)
