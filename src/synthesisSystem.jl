# Finit-dimensional linear operator
synthesize(mtx::Matrix{T}, y) where {T<:Number} = mtx * y

# General Filter banks
synthesize(fb::FilterBank, y; kwargs...) = synthesize(fb, [y], 1; kwargs...)

# Filter bank with polyphase representation
function synthesize(fb::PolyphaseFB{TF,D}, y::Vector{Vector{Array{TY,D}}}, level::Integer; kwargs...) where {TF,TY,D}
    #TODO: リファクタリングする
    #TODO: array2vecblocks を mdarray2polyphaseに置き換える
    nBlocks = [ size(y[l][1]) for l in 1:level ]
    pvy = map(y,nBlocks) do yp, nb
        PolyphaseVector( vcat(
            [ array2vecblocks(ypa, nb).' for ypa in yp ]...
        ), nb)
    end
    synthesize(fb, pvy, level; kwargs...)
end

function synthesize(fb::PolyphaseFB{TF,DF}, y::Vector{Array{TY,DY}}, level::Integer; kwargs...) where {TF,TY,DF,DY}
    if DF != DY-1
        throw(ArgumentError("dimensions of arguments must be satisfy DF + 1 == DY"))
    end

    synthesize(fb, mdarray2polyphase.(y), level; kwargs...)
end

function synthesize(fb::PolyphaseFB{TF,D}, y::Vector{PolyphaseVector{TY,D}}, level::Integer) where {TF,TY,D}
    df = fb.decimationFactor
    function subsynthesize(sy::Vector{PolyphaseVector{TY,D}}, k::Integer)
        ya = if k <= 1
            sy[1]
        else
            dcData = subsynthesize(sy[2:end],k-1)
            dcCoefs = polyphase2mdarray(dcData, df)
            pvdc = mdarray2polyphase(dcCoefs, tuple(fill(1,D)...))
            PolyphaseVector{TY,D}(vcat(pvdc.data, sy[1].data), sy[1].nBlocks)
        end
        multipleSynthesisBank(fb, ya)
    end
    vx = subsynthesize(y, level)
    polyphase2mdarray(vx, df)
end

function multipleSynthesisBank(cc::MDCDL.Cnsolt{TF,D,S}, pvy::PolyphaseVector{TY,D}) where {TF,TY,D,S}
    M = prod(cc.decimationFactor)
    P = cc.nChannels

    uy = concatenateAtoms(cc, PolyphaseVector(cc.symmetry' * pvy.data, pvy.nBlocks))
    y = uy.data

    py = (cc.initMatrices[1] * [ eye(Complex{TF},M) ; zeros(Complex{TF},P-M,M) ])' *  y

    # py .= cc.matrixF' * py
    py .= ctranspose(cc.matrixF) * py

    PolyphaseVector(flipdim(py, 1), pvy.nBlocks)
end

function concatenateAtoms(cc::MDCDL.Cnsolt{TF,D,:TypeI}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    chUpper = 1:fld(cc.nChannels,2)
    chLower = fld(cc.nChannels,2)+1:cc.nChannels

    for d = D:-1:1 # original order
        nShift = -fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        for k = cc.polyphaseOrder[d]:-1:1
            y[chUpper,:] = cc.propMatrices[d][2*k-1]' * y[chUpper,:]
            y[chLower,:] = cc.propMatrices[d][2*k]'   * y[chLower,:]

            B = MDCDL.getMatrixB(cc.nChannels, cc.paramAngles[d][k])
            y .= B' * y
            # y[chLower,:] = circshift(y[chLower,:], (0, nShift))

            if k % 2 == 1
                y[chLower,:] = circshift(y[chLower,:],(0, nShift))
            else
                y[chUpper,:] = circshift(y[chUpper,:],(0, -nShift))
            end
            y .= B * y
        end
        pvy.data .= y
        pvy = MDCDL.ipermutedims(pvy)
    end
    return pvy
end


function concatenateAtoms(cc::MDCDL.Cnsolt{TF,D,:TypeII}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = cc.nChannels
    chEven = 1:P-1

    for d = D:-1:1 # original order
        nShift = -fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        for k = nStages[d]:-1:1
            # second step
            chUpper = 1:cld(P,2)
            chLower = cld(P,2):P

            y[chUpper,:] = cc.propMatrices[d][4*k-1]' * y[chUpper,:]
            y[chLower,:] = cc.propMatrices[d][4*k]'   * y[chLower,:]

            chUpper = 1:fld(P,2)
            chLower = (fld(P,2)+1):(P-1)
            B = MDCDL.getMatrixB(P, cc.paramAngles[d][2*k])
            y[chEven,:] .= B' * y[chEven,:]
            # y[chLower,:] = circshift(y[chLower,:], (0, nShift))
            y[chUpper,:] = circshift(y[chUpper,:], (0, -nShift))
            y[chEven,:] .= B * y[chEven,:]

            # first step

            y[chUpper,:] = cc.propMatrices[d][4*k-3]' * y[chUpper,:]
            y[chLower,:] = cc.propMatrices[d][4*k-2]' * y[chLower,:]

            B = MDCDL.getMatrixB(P, cc.paramAngles[d][2*k-1])
            y[chEven,:] .= B' * y[chEven,:]
            y[chLower,:] = circshift(y[chLower,:], (0, nShift))
            y[chEven,:] .= B * y[chEven,:]
        end
        pvy.data .= y
        pvy = MDCDL.ipermutedims(pvy)
    end
    return pvy
end

function multipleSynthesisBank(cc::MDCDL.Rnsolt{TF,D,S}, pvy::PolyphaseVector{TY,D}) where {TF,TY,D,S}
    M = prod(cc.decimationFactor)
    cM = cld(M,2)
    fM = fld(M,2)
    P = cc.nChannels

    uy = concatenateAtoms(cc, pvy)
    y = uy.data

    W0 = cc.initMatrices[1] * vcat(eye(TF, cM), zeros(TF, P[1] - cM, cM))
    U0 = cc.initMatrices[2] * vcat(eye(TF, fM), zeros(TF, P[2] - fM, fM))
    ty = vcat(W0' * y[1:P[1],:], U0' * y[P[1]+1:end,:])

    ty .= cc.matrixC' * ty

    PolyphaseVector(flipdim(ty, 1), uy.nBlocks)
end

function concatenateAtoms(cc::MDCDL.Rnsolt{TF,D,:TypeI}, pvy::PolyphaseVector{TY,D}) where {TF,TY,D}
    hP = cc.nChannels[1]
    chUpper = 1:hP
    chLower = (1:hP)+hP

    for d = D:-1:1 # original order
        nShift = -fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        for k = cc.polyphaseOrder[d]:-1:1
            y[chLower,:] = cc.propMatrices[d][k]' * y[chLower,:]

            y .= butterfly(y, hP)
            # y[chLower,:] = circshift(y[chLower,:], (0, nShift))
            if isodd(k)
                y[chLower,:] = circshift(y[chLower,:],(0, nShift))
            else
                y[chUpper,:] = circshift(y[chUpper,:],(0, -nShift))
            end
            y .= butterfly(y, hP)
        end
        pvy.data .= y
        pvy = MDCDL.ipermutedims(pvy)
    end
    return pvy
end

function concatenateAtoms(cc::MDCDL.Rnsolt{TF,D,:TypeII}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = sum(cc.nChannels)
    maxP, minP, chMajor, chMinor = if cc.nChannels[1] > cc.nChannels[2]
        (cc.nChannels[1], cc.nChannels[2], 1:cc.nChannels[1], (cc.nChannels[1]+1):P)
    else
        (cc.nChannels[2], cc.nChannels[1], (cc.nChannels[1]+1):P, 1:cc.nChannels[1])
    end

    for d = D:-1:1 # original order
        nShift = -fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        for k = nStages[d]:-1:1
            # second step
            y[chMajor,:] = cc.propMatrices[d][2*k]' * y[chMajor,:]
            y = butterfly(y, minP)
            # y[maxP+1:end,:] = circshift(y[maxP+1:end,:], (0, nShift))
            y[1:maxP,:] = circshift(y[1:maxP,:], (0, -nShift))
            y = butterfly(y, minP)

            # first step
            y[chMinor,:] = cc.propMatrices[d][2*k-1]' * y[chMinor,:]

            y = butterfly(y, minP)
            y[minP+1:end,:] = circshift(y[minP+1:end,:], (0, nShift))
            y = butterfly(y, minP)
        end
        pvy.data .= y
        pvy = MDCDL.ipermutedims(pvy)
    end
    return pvy
end

function synthesize(pfb::MDCDL.ParallelFB{TF,D}, y::Vector{Vector{Array{TY,D}}}, level::Integer) where {TF,TY,D}
    df = pfb.decimationFactor
    function subsynthesize(sy::Vector{Vector{Array{TY,D}}}, k::Integer)
        ya = if k <= 1
            sy[1]
        else
            [ subsynthesize(sy[2:end],k-1), sy[1]... ]
        end
        sxs = map(ya, pfb.synthesisFilters) do yp, sfp
            MDCDL.mdfilter(MDCDL.upsample(yp, df), sfp; operation=:conv)
        end
        circshift(sum(sxs), -1 .* df .* cld.(pfb.polyphaseOrder,2))
        # sum(sxs)
    end
    vx = subsynthesize(y, level)
end
