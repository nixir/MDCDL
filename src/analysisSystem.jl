function analyze(fb::MDCDL.PolyphaseFB{TF,D}, x::Array{TX,D}, level::Integer; kwargs...) where {TF,TX,D}
    analyze(fb, mdarray2polyphase(x, fb.decimationFactor), level; kwargs...)
end

function analyze(fb::MDCDL.PolyphaseFB{TF,D}, x::PolyphaseVector{TX,D}, level::Integer = 1; outputMode=:spacial) where {TF,TX,D}
    function subanalyze(sx::PolyphaseVector{TS,D}, k::Integer) where TS
        sy = multipleAnalysisBank(fb, sx)
        if k <= 1
            return [ sy ]
        else
            res = Vector{PolyphaseVector{TF,D}}(k)
            res[1] = PolyphaseVector{TF,D}(sy.data[2:end,:], sy.nBlocks)

            # reshape Low-pass MD filter
            dcCoefs = PolyphaseVector{TF,D}(sy.data[1:1,:], sy.nBlocks)
            dcData = polyphase2mdarray(dcCoefs, tuple(fill(1,D)...))
            nsx = mdarray2polyphase(dcData, fb.decimationFactor)

            res[2:end] = subanalyze(nsx, k-1)
            return res
        end
    end

    y = subanalyze(x,level)

    if outputMode == :polyphase
        y
    elseif outputMode == :spacial
        [ [ MDCDL.vecblocks2array(py.data[p,:], py.nBlocks, tuple(ones(Integer,D)...)) for p in 1:size(py.data,1) ] for py in y]
    end
end

function multipleAnalysisBank(cc::MDCDL.Cnsolt{D,S,TF}, pvx::PolyphaseVector{TX,D}) where {TF,TX,D,S}
    const M = prod(cc.decimationFactor)
    const P = cc.nChannels

    x = pvx.data
    tx = cc.matrixF * flipdim(x, 1)

    const V0 = cc.initMatrices[1] * [ eye(Complex{TF},M) ; zeros(Complex{TF},P-M,M) ]
    ux = V0 * tx

    extx = extendAtoms(cc, PolyphaseVector(ux,pvx.nBlocks))
    PolyphaseVector(cc.symmetry*extx.data, extx.nBlocks)
end

function extendAtoms(cc::MDCDL.Cnsolt{D,1,TF}, pvx::PolyphaseVector{TX,D}; boundary=:circular) where {TF,TX,D}
    const chUpper = 1:fld(cc.nChannels,2)
    const chLower = fld(cc.nChannels,2)+1:cc.nChannels

    for d = 1:D # original order
        const nShift = fld(size(pvx,2), pvx.nBlocks[1])
        pvx = MDCDL.permutedims(pvx)
        x = pvx.data
        for k = 1:cc.polyphaseOrder[d]
            B = MDCDL.getMatrixB(cc.nChannels, cc.paramAngles[d][k])

            x .= B' * x
            x[chLower,:] = circshift(x[chLower,:], (0, nShift))
            # if k % 2 == 1
            #     x[chLower,:] = circshift(x[chLower,:],(0, nShift))
            # else
            #     x[chUpper,:] = circshift(x[chUpper,:],(0, -nShift))
            # end
            x .= B * x

            x[chUpper,:] = cc.propMatrices[d][2*k-1] * x[chUpper,:]
            x[chLower,:] = cc.propMatrices[d][2*k]   * x[chLower,:]
        end
        pvx.data .= x
    end
    return pvx
end

function extendAtoms(cc::MDCDL.Cnsolt{D,2,TF}, pvx::PolyphaseVector{TX,D}; boundary=:circular) where {TF,TX,D}
    const nStages = fld.(cc.polyphaseOrder,2)
    const P = cc.nChannels
    const chEven = 1:P-1

    for d = 1:D # original order
        const nShift = fld(size(pvx,2), pvx.nBlocks[1])
        pvx = MDCDL.permutedims(pvx)
        x = pvx.data
        for k = 1:nStages[d]
            # first step
            chUpper = 1:fld(P,2)
            chLower = (fld(P,2)+1):(P-1)
            B = MDCDL.getMatrixB(P, cc.paramAngles[d][2*k-1])

            x[chEven,:]= B' * x[chEven,:]
            x[chLower,:] = circshift(x[chLower,:], (0, nShift))
            # if k % 2 == 1
            #     x[chLower,:] = circshift(x[chLower,:],(0, nShift))
            # else
            #     x[chUpper,:] = circshift(x[chUpper,:],(0, -nShift))
            # end
            x[chEven,:] .= B * x[chEven,:]

            x[chUpper,:] = cc.propMatrices[d][4*k-3] * x[chUpper,:]
            x[chLower,:] = cc.propMatrices[d][4*k-2] * x[chLower,:]

            # second step
            chUpper = 1:cld(P,2)
            chLower = cld(P,2):P

            B = MDCDL.getMatrixB(P, cc.paramAngles[d][2*k])

            x[chEven,:] .= B' * x[chEven,:]
            x[chLower,:] = circshift(x[chLower,:], (0, nShift))
            x[chEven,:] .= B * x[chEven,:]

            x[chLower,:] = cc.propMatrices[d][4*k]   * x[chLower,:]
            x[chUpper,:] = cc.propMatrices[d][4*k-1] * x[chUpper,:]
        end
        pvx.data .= x
    end
    return pvx
end

function multipleAnalysisBank(cc::MDCDL.Rnsolt{D,S,TF}, pvx::PolyphaseVector{TX,D}) where {TF,TX,D,S}
    const M = prod(cc.decimationFactor)
    const cM = cld(M,2)
    const fM = fld(M,2)
    const P = cc.nChannels
    x = pvx.data

    tx = cc.matrixC * flipdim(x, 1)

    # const V0 = cc.initMatrices[1] * [ eye(T,M) ; zeros(T,P-M,M) ]
    const W0 = cc.initMatrices[1] * vcat(eye(TF, cM), zeros(TF, P[1] - cM, cM))
    const U0 = cc.initMatrices[2] * vcat(eye(TF, fM), zeros(TF, P[2] - fM, fM))

    ux = PolyphaseVector(vcat(W0 * tx[1:cM, :], U0 * tx[cM+1:end, :]), pvx.nBlocks)

    extendAtoms(cc, ux)
end

function extendAtoms(cc::MDCDL.Rnsolt{D,1,TF}, px::PolyphaseVector{TX,D}, boundary=:circular) where {TF,TX,D}
    const hP = cc.nChannels[1]
    # const chUpper = 1:P
    const chLower = (1:hP)+hP

    for d = 1:D # original order
        const nShift = fld(size(px,2), px.nBlocks[1])
        px = MDCDL.permutedims(px)
        for k = 1:cc.polyphaseOrder[d]

            px .= butterfly(px.data, hP)

            px[chLower,:] = circshift(px[chLower,:], (0, nShift))
            # if k % 2 == 1
            #     px[rngLower...] = circshift(px[rngLower...],(0, nShifts[d]))
            # else
            #     px[rngUpper...] = circshift(px[rngUpper...],(0, -nShifts[d]))
            # end
            px .= butterfly(px.data, hP)

            px[chLower,:] = cc.propMatrices[d][k] * px[chLower,:]
        end
    end
    return px
end

function extendAtoms(cc::MDCDL.Rnsolt{D,2,TF}, pvx::PolyphaseVector{TX,D}; boundary=:circular) where {TF,TX,D}
    const nStages = fld.(cc.polyphaseOrder,2)
    const P = sum(cc.nChannels)
    const maxP, minP, chMajor, chMinor = if cc.nChannels[1] > cc.nChannels[2]
        (cc.nChannels[1], cc.nChannels[2], 1:cc.nChannels[1], (cc.nChannels[1]+1):P)
    else
        (cc.nChannels[2], cc.nChannels[1], (cc.nChannels[1]+1):P, 1:cc.nChannels[1])
    end

    for d = 1:D # original order
        const nShift = fld(size(pvx,2), pvx.nBlocks[1])
        pvx = MDCDL.permutedims(pvx)
        x = pvx.data
        for k = 1:nStages[d]
            # first step
            x = butterfly(x, minP)
            x[minP+1:end,:] = circshift(x[minP+1:end,:], (0, nShift))
            # if k % 2 == 1
            #     x[chLower,:] = circshift(x[chLower,:],(0, nShift))
            # else
            #     x[chUpper,:] = circshift(x[chUpper,:],(0, -nShift))
            # end
            x = butterfly(x, minP)

            x[chMinor,:] = cc.propMatrices[d][2*k-1] * x[chMinor,:]

            # second step
            x = butterfly(x, minP)
            x[maxP+1:end,:] = circshift(x[maxP+1:end,:], (0, nShift))
            x = butterfly(x, minP)

            x[chMajor,:] = cc.propMatrices[d][2*k]   * x[chMajor,:]
        end
        pvx.data .= x
    end
    return pvx
end

function analyze(pfb::MDCDL.ParallelFB{TF,D}, x::Array{TX,D}, level::Integer = 1) where {TF,TX,D}
    const df = pfb.decimationFactor
    const offset = df .- 1

    function subanalyze(sx::Array{TS,D}, k::Integer) where TS
        sy = [  circshift(MDCDL.downsample(MDCDL.mdfilter(sx, f; operation=:conv), df, offset), 0) for f in pfb.analysisFilters ]
        if k <= 1
            return [ sy ]
        else
            res = Vector{Vector{Array{TF,D}}}(k)
            res[1] = sy[2:end]
            res[2:end] = subanalyze(sy[1], k-1)
            return res
        end
    end

    subanalyze(x,level)
end
