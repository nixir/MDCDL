function synthesize(fb::MDCDL.FilterBank{T,D}, y::Vector{Vector{Array{TY,D}}}, scales::Vector{NTuple{D,Int}}, level::Integer = 1) where {T,TY,D}
    function subsynthesize(sy::Vector{Vector{Array{TY,D}}}, k::Integer)
        ya = if k <= 1
            sy[1]
        else
            [ subsynthesize(sy[2:end],k-1), sy[1]... ]
        end
        stepSynthesisBank(fb, ya, scales[level-k+1])
    end
    vx = subsynthesize(y, level)
    reshape(vx, scales[1]...)
end

function stepSynthesisBank(fb::MDCDL.PolyphaseFB{TF,D}, y::Array{TY}, szRegion::NTuple{D}; inputMode=:normal) where {TF,TY,D}
    const M = prod(fb.decimationFactor)
    const P = sum(fb.nChannels)
    const nBlocks = fld.(szRegion, fb.decimationFactor)

    py = if inputMode == :normal
        PolyphaseVector(vcat([MDCDL.array2vecblocks(y[p], nBlocks).' for p in 1:P]...), nBlocks)
    elseif inputMode == :augumented
        mdarray2polyphase(y)
    else
        throw(ArgumentError("inputMode=$inputMode: This option is not available."))
    end

    size(py.data)

    px = multipleSynthesisPolyphaseMat(fb, py)
    polyphase2mdarray(px, fb.decimationFactor)

end


function stepSynthesisBank(fb::MDCDL.PolyphaseFB{TF,DF}, y::Array{TY,DY}; kwargs...) where {TF,DF,TY,DY}
    if DY != DF + 1
        throw(DimensionMismatch("Dimension mismatch. Dim. of the filter bank = $DF, Dim. of the input = $DY."))
    end
    szRegion = size(y)[1:DY-1] .* fb.decimationFactor
    stepSynthesisBank(fb,y,szRegion; kwargs...)
end

function multipleSynthesisPolyphaseMat(cc::MDCDL.Cnsolt{D,S,TF}, pvy::PolyphaseVector{TY,D}) where {TF,TY,D,S}
    const M = prod(cc.decimationFactor)
    const P = cc.nChannels

    uy = concatenateAtoms(cc, PolyphaseVector(cc.symmetry' * pvy.data, pvy.nBlocks))
    y = uy.data

    py = (cc.initMatrices[1] * [ eye(Complex{TF},M) ; zeros(Complex{TF},P-M,M) ])' *  y

    # py .= cc.matrixF' * py
    py .= ctranspose(cc.matrixF) * py

    PolyphaseVector(flipdim(py, 1), pvy.nBlocks)
end

function concatenateAtoms(cc::MDCDL.Cnsolt{D,1,TF}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    const chUpper = 1:fld(cc.nChannels,2)
    const chLower = fld(cc.nChannels,2)+1:cc.nChannels

    for d = D:-1:1 # original order
        const nShift = -fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        for k = cc.polyphaseOrder[d]:-1:1
            y[chUpper,:] = cc.propMatrices[d][2*k-1]' * y[chUpper,:]
            y[chLower,:] = cc.propMatrices[d][2*k]'   * y[chLower,:]

            B = MDCDL.getMatrixB(cc.nChannels, cc.paramAngles[d][k])
            y .= B' * y
            y[chLower,:] = circshift(y[chLower,:], (0, nShift))

            # if k % 2 == 1
            #     y[chLower,:] = circshift(y[chLower,:],(0, nShift))
            # else
            #     y[chUpper,:] = circshift(y[chUpper,:],(0, -nShift))
            # end
            y .= B * y
        end
        pvy.data .= y
        pvy = MDCDL.ipermutedims(pvy)
    end
    return pvy
end


function concatenateAtoms(cc::MDCDL.Cnsolt{D,2,TF}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    const nStages = fld.(cc.polyphaseOrder,2)
    const P = cc.nChannels
    const chEven = 1:P-1

    for d = D:-1:1 # original order
        const nShift = -fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        for k = nStages[d]:-1:1
            # second step
            chUpper = 1:cld(P,2)
            chLower = cld(P,2):P

            y[chUpper,:] = cc.propMatrices[d][4*k-1]' * y[chUpper,:]
            y[chLower,:] = cc.propMatrices[d][4*k]'   * y[chLower,:]

            B = MDCDL.getMatrixB(P, cc.paramAngles[d][2*k])
            y[chEven,:] .= B' * y[chEven,:]
            y[chLower,:] = circshift(y[chLower,:], (0, nShift))

            # if k % 2 == 1
            #     y[chLower,:] = circshift(y[chLower,:],(0, nShift))
            # else
            #     y[chUpper,:] = circshift(y[chUpper,:],(0, -nShift))
            # end
            y[chEven,:] .= B * y[chEven,:]

            # first step

            chUpper = 1:fld(P,2)
            chLower = (fld(P,2)+1):(P-1)

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

function multipleSynthesisPolyphaseMat(cc::MDCDL.Rnsolt{D,S,TF}, pvy::PolyphaseVector{TY,D}) where {TF,TY,D,S}
    const M = prod(cc.decimationFactor)
    const cM = cld(M,2)
    const fM = fld(M,2)
    const P = cc.nChannels

    uy = concatenateAtoms(cc, pvy)
    y = uy.data

    const W0 = cc.initMatrices[1] * vcat(eye(TF, cM), zeros(TF, P[1] - cM, cM))
    const U0 = cc.initMatrices[2] * vcat(eye(TF, fM), zeros(TF, P[2] - fM, fM))
    ty = vcat(W0' * y[1:P[1],:], U0' * y[P[1]+1:end,:])

    ty .= cc.matrixC' * ty

    PolyphaseVector(flipdim(ty, 1), uy.nBlocks)
end

function concatenateAtoms(cc::MDCDL.Rnsolt{D,1,TF}, pvy::PolyphaseVector{TY,D}) where {TF,TY,D}
    const P = cc.nChannels[1]
    # const chUpper = 1:P
    const chLower = (1:P)+P

    for d = D:-1:1 # original order
        const nShift = -fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        for k = cc.polyphaseOrder[d]:-1:1
            y[chLower,:] = cc.propMatrices[d][k]' * y[chLower,:]

            y .= butterfly(y, P)
            y[chLower,:] = circshift(y[chLower,:], (0, nShift))
            # if k % 2 == 1
            #     py[rngLower...] = circshift(py[rngLower...],(0, nShifts[d]))
            # else
            #     py[rngUpper...] = circshift(py[rngUpper...],(0, nShifts[d]))
            # end
            y .= butterfly(y, P)
        end
        pvy.data .= y
        pvy = MDCDL.ipermutedims(pvy)
    end
    return pvy
end

function concatenateAtoms(cc::MDCDL.Rnsolt{D,2,TF}, pvy::PolyphaseVector{TY,D}; boundary=:circular) where {TF,TY,D}
    const nStages = fld.(cc.polyphaseOrder,2)
    const P = sum(cc.nChannels)
    const maxP, minP, chMajor, chMinor = if cc.nChannels[1] > cc.nChannels[2]
        (cc.nChannels[1], cc.nChannels[2], 1:cc.nChannels[1], (cc.nChannels[1]+1):P)
    else
        (cc.nChannels[2], cc.nChannels[1], (cc.nChannels[1]+1):P, 1:cc.nChannels[1])
    end

    for d = D:-1:1 # original order
        const nShift = -fld(size(pvy,2), pvy.nBlocks[end])
        y = pvy.data
        for k = nStages[d]:-1:1
            # second step
            y[chMajor,:] = cc.propMatrices[d][2*k]' * y[chMajor,:]
            y = butterfly(y, minP)
            y[maxP+1:end,:] = circshift(y[maxP+1:end,:], (0, nShift))

            # if k % 2 == 1
            #     y[chLower,:] = circshift(y[chLower,:],(0, nShift))
            # else
            #     y[chUpper,:] = circshift(y[chUpper,:],(0, -nShift))
            # end
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

function stepSynthesisBank(pfb::MDCDL.ParallelFB{TF,D}, y::Vector{Array{TY,D}}, szRegion) where {TF,TY,D}
    const df = pfb.decimationFactor
    sxs = map(y, pfb.synthesisFilters) do yp, sfp
        MDCDL.mdfilter(MDCDL.upsample(yp, df), sfp; operation=:conv)
    end
    circshift(sum(sxs), -1 .* df .* pfb.polyphaseOrder)
end
