using ImageFiltering
using OffsetArrays
# Finit-dimensional linear operator
synthesize(mtx::Matrix{T}, y) where {T<:Number} = mtx * y

# Filter bank with polyphase representation
# outputMode = :reshaped
function synthesize(fb::PolyphaseFB{TF,D}, y::Vector{Array{TY,D}}; kwargs...) where {TF,TY,D}
    nBlocks = size(y[1])
    # if any(size.(y) .!= nBlocks)
    #     throw(ArgumentError("size Error"))
    # end
    pvy = PolyphaseVector( transpose(hcat(map(vec, y)...)), nBlocks)
    # end
    synthesize(fb, pvy; kwargs...)
end

# outputMode = :augumented
function synthesize(fb::PolyphaseFB{TF,DF}, y::Array{TY,DY}; kwargs...) where {TF,TY,DF,DY}
    if DF != DY-1
        throw(ArgumentError("dimensions of arguments must be satisfy DF + 1 == DY"))
    end

    synthesize(fb, mdarray2polyphase(y); kwargs...)
end

# outputMode = :vector
function synthesize(fb::PolyphaseFB{TF,D}, y::Vector{TY}, szdata::NTuple{D}; kwargs...) where {TF, TY, D}
    yaug = reshape(y, fld(szdata, fb.decimationFactor)..., p);
    synthesize(fb, yaug; kwargs...)
end

# outputMode = :polyphase
function synthesize(fb::PolyphaseFB{TF,D}, y::PolyphaseVector{TY,D}) where {TF,TY,D}
    df = fb.decimationFactor
    vx = multipleSynthesisBank(fb, y)
    polyphase2mdarray(vx, df)
end

function multipleSynthesisBank(cc::Cnsolt{TF,D,S}, pvy::PolyphaseVector{TY,D}) where {TF,TY,D,S}
    M = prod(cc.decimationFactor)
    P = cc.nChannels

    uy = concatenateAtoms(cc, PolyphaseVector(cc.symmetry' * pvy.data, pvy.nBlocks))

    py = (cc.initMatrices[1] * eye(Complex{TF},P,M))' * uy.data

    py .= ctranspose(cc.matrixF) * py

    PolyphaseVector(flipdim(py, 1), pvy.nBlocks)
end

function concatenateAtoms(cc::Cnsolt{TF,D,:TypeI}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    P = cc.nChannels

    for d = D:-1:1
        nShift = fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        # submatrices
        yu = view(y, 1:fld(P,2), :)
        yl = view(y, (fld(P,2)+1):P, :)
        for k = cc.polyphaseOrder[d]:-1:1
            yu .= cc.propMatrices[d][2*k-1]' * yu
            yl .= cc.propMatrices[d][2*k]'   * yl

            B = getMatrixB(P, cc.paramAngles[d][k])
            y .= B' * y

            if isodd(k)
                yl .= circshift(yl, (0, -nShift))
            else
                yu .= circshift(yu, (0, nShift))
            end
            y .= B * y
        end
        pvy.data .= y
        pvy = ipermutedims(pvy)
    end
    return pvy
end


function concatenateAtoms(cc::Cnsolt{TF,D,:TypeII}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = cc.nChannels
    chEven = 1:P-1

    for d = D:-1:1
        nShift = fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        # submatrices
        ye  = view(y, 1:P-1, :)
        yu1 = view(y, 1:fld(P,2), :)
        yl1 = view(y, (fld(P,2)+1):(P-1), :)
        yu2 = view(y, 1:cld(P,2), :)
        yl2 = view(y, cld(P,2):P, :)
        for k = nStages[d]:-1:1
            # second step

            yu2 .= cc.propMatrices[d][4*k-1]' * yu2
            yl2 .= cc.propMatrices[d][4*k]'   * yl2

            B = getMatrixB(P, cc.paramAngles[d][2*k])
            ye  .= B' * ye
            yu1 .= circshift(yu1, (0, nShift))
            ye  .= B * ye

            # first step

            yu1 .= cc.propMatrices[d][4*k-3]' * yu1
            yl1 .= cc.propMatrices[d][4*k-2]' * yl1

            B = getMatrixB(P, cc.paramAngles[d][2*k-1])
            ye  .= B' * ye
            yl1 .= circshift(yl1, (0, -nShift))
            ye  .= B * ye
        end
        pvy.data .= y
        pvy = ipermutedims(pvy)
    end
    return pvy
end

function multipleSynthesisBank(cc::Rnsolt{TF,D,S}, pvy::PolyphaseVector{TY,D}) where {TF,TY,D,S}
    M = prod(cc.decimationFactor)
    cM = cld(M,2)
    fM = fld(M,2)
    P = cc.nChannels

    uy = concatenateAtoms(cc, pvy)
    y = uy.data

    W0 = cc.initMatrices[1] * eye(TF, P[1], cM)
    U0 = cc.initMatrices[2] * eye(TF, P[2], fM)
    ty = vcat(W0' * y[1:P[1],:], U0' * y[P[1]+1:end,:])

    ty .= cc.matrixC' * ty

    PolyphaseVector(flipdim(ty, 1), uy.nBlocks)
end

function concatenateAtoms(cc::Rnsolt{TF,D,:TypeI}, pvy::PolyphaseVector{TY,D}) where {TF,TY,D}
    hP = cc.nChannels[1]

    for d = D:-1:1
        nShift = fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        # submatrices
        yu = view(y, 1:hP, :)
        yl = view(y, (1:hP)+hP, :)
        for k = cc.polyphaseOrder[d]:-1:1
            yl .= cc.propMatrices[d][k]' * yl

            y  .= butterfly(y, hP)
            if isodd(k)
                yl .= circshift(yl, (0, -nShift))
            else
                yu .= circshift(yu, (0, nShift))
            end
            y .= butterfly(y, hP)
        end
        pvy.data .= y
        pvy = ipermutedims(pvy)
    end
    return pvy
end

function concatenateAtoms(cc::Rnsolt{TF,D,:TypeII}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = sum(cc.nChannels)
    maxP, minP, chMajor, chMinor = if cc.nChannels[1] > cc.nChannels[2]
        (cc.nChannels[1], cc.nChannels[2], 1:cc.nChannels[1], (cc.nChannels[1]+1):P)
    else
        (cc.nChannels[2], cc.nChannels[1], (cc.nChannels[1]+1):P, 1:cc.nChannels[1])
    end

    for d = D:-1:1
        nShift = fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        # submatrices
        ys1 = view(y, minP+1:P, :)
        ys2 = view(y, 1:maxP, :)
        ymj = view(y, chMajor, :)
        ymn = view(y, chMinor, :)
        for k = nStages[d]:-1:1
            # second step
            ymj .= cc.propMatrices[d][2*k]' * ymj

            y   .= butterfly(y, minP)
            ys2 .= circshift(ys2, (0, nShift))
            y   .= butterfly(y, minP)

            # first step
            ymn .= cc.propMatrices[d][2*k-1]' * ymn

            y   .= butterfly(y, minP)
            ys1 .= circshift(ys1, (0, -nShift))
            y   .= butterfly(y, minP)
        end
        pvy.data .= y
        pvy = ipermutedims(pvy)
    end
    return pvy
end

function synthesize(pfb::ParallelFB{TF,D}, y::Vector{Array{TY,D}}) where {TF,TY,D}
    df = pfb.decimationFactor
    ord = pfb.polyphaseOrder
    region = colon.(1,df.*(ord.+1)) .- df.*cld.(ord,2) .- 1

    sxs = map(y, pfb.synthesisFilters) do yp, sfp
        upimg = upsample(yp, df)
        ker = reflect(OffsetArray(sfp, region...))
        imfilter(upimg, ker, "circular" , ImageFiltering.FIR())
    end
    sum(sxs)
end

# outputMode= :reshaped
function synthesize(msfb::Multiscale{TF,D}, y::Vector{Vector{Array{TY,D}}}) where {TF,TY,D}
    function subsynthesize(sy::Vector, k::Integer)
        ya = if k <= 1
            sy[1]
        else
            [ subsynthesize(sy[2:end], k-1), sy[1]... ]
        end
        synthesize(msfb.filterBank, ya)
    end
    subsynthesize(y, msfb.treeLevel)
end

# outputMode = :augumented
function synthesize(msfb::Multiscale{TF,DF}, y::Vector{Array{TY,DY}}) where {TF,TY,DF,DY}
    if DF != DY-1
        throw(ArgumentError("dimensions of arguments must be satisfy DF + 1 == DY"))
    end
    yrd = map(y) do ys
        [ ys[fill(:, DF)...,p] for p in 1:size(ys, DY) ]
    end
    synthesize(msfb, yrd)
end

# outputMode = :vector
function synthesize(msfb::Multiscale{TF,D}, y::Vector{TY}, szdata::NTuple{D}) where {TF,TY,D}
    df = msfb.filterBank.decimationFactor
    P = sum(msfb.filterBank.nChannels)
    nCoefs = [ prod(fld.(szdata, df.^l)) * (l==msfb.treeLevel ? P : P-1) for l in 1:msfb.treeLevel ]

    augCoefs = map(1:msfb.treeLevel) do l
        rang = (1:nCoefs[l]) + sum(nCoefs[1:l-1])
        reshape(y[rang], fld.(szdata, df.^l)..., (l==msfb.treeLevel ? P : P-1))
    end
    synthesize(msfb, augCoefs)
end
