using ImageFiltering: imfilter, reflect, FIR
using OffsetArrays: OffsetArray

# import Base.*

function synthesize(syn::AbstractSynthesizer{TF,D}, y::AbstractArray) where {TF,D}
    pvy = if syn.shape == :normal
        nBlocks = size(y[1])
        PolyphaseVector( Matrix(transpose(hcat(map(vec, y)...))), nBlocks)
    elseif syn.shape == :augumented
        mdarray2polyphase(y)
    elseif syn.shape == :vector
        ty = reshape(y, fld.(syn.datasize, syn.codebook.decimationFactor)..., sum(syn.codebook.nChannels))
        mdarray2polyphase(ty)
    else
        error("Invalid argument.")
    end

    pvx = synthesize(syn.codebook, pvy)
    polyphase2mdarray(pvx, syn.codebook.decimationFactor)
end
(syn::NsoltSynthesizer)(y::AbstractArray) = synthesize(syn, y)

function synthesize(cc::Cnsolt{TF,D,S}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D,S}
    M = prod(cc.decimationFactor)
    P = cc.nChannels

    uy = concatenateAtoms!(cc, PolyphaseVector(cc.symmetry' * pvy.data, pvy.nBlocks); kwargs...)

    py = (cc.initMatrices[1] * Matrix{Complex{TF}}(I,P,M))' * uy.data
    py .= reverse(cc.matrixF, dims=2)' * py

    PolyphaseVector(py, pvy.nBlocks)
end

function concatenateAtoms!(cc::Cnsolt{TF,D,:TypeI}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    P = cc.nChannels

    for d = D:-1:1
        nShift = fld(size(pvy.data,2), pvy.nBlocks[end])
        # submatrices
        y  = view(pvy.data, :, :)
        yu = view(pvy.data, 1:fld(P,2), :)
        yl = view(pvy.data, (fld(P,2)+1):P, :)
        for k = cc.polyphaseOrder[d]:-1:1
            yu .= cc.propMatrices[d][2*k-1]' * yu
            yl .= cc.propMatrices[d][2*k]'   * yl

            B = getMatrixB(P, cc.paramAngles[d][k])
            y .= B' * y

            if isodd(k)
                # yl .= circshift(yl, (0, -nShift))
                shiftBackward!(Val{boundary}, yl, nShift)
            else
                # yu .= circshift(yu, (0, nShift))
                shiftForward!(Val{boundary}, yu, nShift)
            end
            y .= B * y
        end
        pvy = ipermutedims(pvy)
    end
    return pvy
end


function concatenateAtoms!(cc::Cnsolt{TF,D,:TypeII}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = cc.nChannels
    chEven = 1:(P-1)

    for d = D:-1:1
        nShift = fld(size(pvy.data,2), pvy.nBlocks[end])
        # submatrices
        ye  = view(pvy.data, 1:(P-1), :)
        yu1 = view(pvy.data, 1:fld(P,2), :)
        yl1 = view(pvy.data, (fld(P,2)+1):(P-1), :)
        yu2 = view(pvy.data, 1:cld(P,2), :)
        yl2 = view(pvy.data, cld(P,2):P, :)
        for k = nStages[d]:-1:1
            # second step

            yu2 .= cc.propMatrices[d][4*k-1]' * yu2
            yl2 .= cc.propMatrices[d][4*k]'   * yl2

            B = getMatrixB(P, cc.paramAngles[d][2*k])
            ye  .= B' * ye
            # yu1 .= circshift(yu1, (0, nShift))
            shiftForward!(Val{boundary}, yu1, nShift)
            ye  .= B * ye

            # first step

            yu1 .= cc.propMatrices[d][4*k-3]' * yu1
            yl1 .= cc.propMatrices[d][4*k-2]' * yl1

            B = getMatrixB(P, cc.paramAngles[d][2*k-1])
            ye  .= B' * ye
            # yl1 .= circshift(yl1, (0, -nShift))
            shiftBackward!(Val{boundary}, yl1, nShift)
            ye  .= B * ye
        end
        pvy = ipermutedims(pvy)
    end
    return pvy
end

function synthesize(cc::Rnsolt{TF,D,S}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D,S}
    M = prod(cc.decimationFactor)
    cM = cld(M,2)
    fM = fld(M,2)
    nch = cc.nChannels

    cpvy = deepcopy(pvy)
    uy = concatenateAtoms!(cc, cpvy; kwargs...)
    y = uy.data

    W0 = cc.initMatrices[1] * Matrix{TF}(I, nch[1], cM)
    U0 = cc.initMatrices[2] * Matrix{TF}(I, nch[2], fM)
    ty = vcat(W0' * y[1:nch[1],:], U0' * y[(nch[1]+1):end,:])
    ty .= reverse(cc.matrixC, dims=2)' * ty

    PolyphaseVector(ty, uy.nBlocks)
end

function concatenateAtoms!(cc::Rnsolt{TF,D,:TypeI}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    hP = cc.nChannels[1]

    for d = D:-1:1
        nShift = fld(size(pvy.data,2), pvy.nBlocks[end])
        # submatrices
        yu = view(pvy.data, 1:hP, :)
        yl = view(pvy.data, (1:hP) .+ hP, :)
        for k = cc.polyphaseOrder[d]:-1:1
            yl .= cc.propMatrices[d][k]' * yl

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl
            if isodd(k)
                shiftBackward!(Val{boundary}, yl, nShift)
            else
                shiftForward!(Val{boundary}, yu, nShift)
            end
            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl
        end
        pvy = ipermutedims(pvy)
    end
    return pvy
end

function concatenateAtoms!(cc::Rnsolt{TF,D,:TypeII}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = sum(cc.nChannels)
    maxP, minP, chMajor, chMinor = if cc.nChannels[1] > cc.nChannels[2]
        (cc.nChannels[1], cc.nChannels[2], 1:cc.nChannels[1], (cc.nChannels[1]+1):P)
    else
        (cc.nChannels[2], cc.nChannels[1], (cc.nChannels[1]+1):P, 1:cc.nChannels[1])
    end

    for d = D:-1:1
        nShift = fld(size(pvy.data,2), pvy.nBlocks[end])
        # submatrices
        yu  = view(pvy.data, 1:minP, :)
        yl  = view(pvy.data, (P-minP+1):P, :)
        ys1 = view(pvy.data, (minP+1):P, :)
        ys2 = view(pvy.data, 1:maxP, :)
        ymj = view(pvy.data, chMajor, :)
        ymn = view(pvy.data, chMinor, :)
        for k = nStages[d]:-1:1
            # second step
            ymj .= cc.propMatrices[d][2*k]' * ymj

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl

            shiftForward!(Val{boundary}, ys2, nShift)

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl

            # first step
            ymn .= cc.propMatrices[d][2*k-1]' * ymn

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl

            shiftBackward!(Val{boundary}, ys1, nShift)

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl
        end
        pvy = ipermutedims(pvy)
    end
    return pvy
end

# function synthesize(pfb::ParallelFB{TF,D}, y::AbstractVector{AbstractArray{TY,D}}; alg=FIR()) where {TF,TY,D}
function synthesize(pfb::ParallelFB{TF,D}, y::AbstractVector{Array{TY,D}}; alg=FIR()) where {TF,TY,D}
    df = pfb.decimationFactor
    ord = pfb.polyphaseOrder

    nShift = df .* cld.(ord, 2) .+ 1
    region = UnitRange.(1 .- nShift, df .* (ord .+ 1) .- nShift)

    sxs = map(y, pfb.synthesisFilters) do yp, sfp
        upimg = upsample(yp, df)
        ker = reflect(OffsetArray(sfp, region...))
        imfilter(upimg, ker, "circular", alg)
    end
    sum(sxs)
end

# shape= :normal
function synthesize(msfb::Multiscale{TF,D}, y::AbstractVector{Vector{Array{TY,D}}}) where {TF,TY,D}
    subsynthesize(msfb.filterBank, y, msfb.treeLevel)
end

function subsynthesize(fb::FilterBank, sy::AbstractVector, k::Integer)
    ya = if k <= 1
        sy[1]
    else
        [ subsynthesize(fb, sy[2:end], k-1), sy[1]... ]
    end
    synthesize(fb, ya)
end

# shape = :augumented
function synthesize(msfb::Multiscale{TF,DF}, y::AbstractVector{Array{TY,DY}}) where {TF,TY,DF,DY}
    if DF != DY-1
        throw(ArgumentError("dimensions of arguments must be satisfy DF + 1 == DY"))
    end
    yrd = map(y) do ys
        [ ys[fill(:, DF)...,p] for p in 1:size(ys, DY) ]
    end
    synthesize(msfb, yrd)
end

# shape = :vector
function synthesize(msfb::Multiscale{TF,D}, y::AbstractVector{TY}, szdata::NTuple{D}) where {TF,TY,D}
    df = msfb.filterBank.decimationFactor
    P = sum(msfb.filterBank.nChannels)
    nCoefs = [ prod(fld.(szdata, df.^l)) * (l==msfb.treeLevel ? P : P-1) for l in 1:msfb.treeLevel ]

    augCoefs = map(1:msfb.treeLevel) do l
        rang = (1:nCoefs[l]) .+ sum(nCoefs[1:(l-1)])
        reshape(y[rang], fld.(szdata, df.^l)..., (l==msfb.treeLevel ? P : P-1))
    end
    synthesize(msfb, augCoefs)
end
