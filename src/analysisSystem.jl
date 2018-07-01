function analyze(fb::MDCDL.FilterBank{TF,D}, x::Array{TX,D}, level::Integer = 1) where {TF,TX,D}
    function subanalyze(sx::Array{TS,D}, k::Integer) where TS
        sy = stepAnalysisBank(fb, sx)
        if k <= 1
            return [ sy ]
        else
            res = Array{Array{Array{TF,D},1},1}(k)
            res[1] = sy[2:end]
            res[2:end] = subanalyze(sy[1], k-1)
            return res
        end
    end

    scales = [ fld.(size(x), fb.decimationFactor.^l) for l in 0:level]
    ( subanalyze(x,level), scales )
end

function stepAnalysisBank(fb::MDCDL.PolyphaseFB{TF,D}, x::Array{TX,D}; outputMode=:normal) where {TF,TX,D}
    const df = fb.decimationFactor
    const M = prod(df)

    px = MDCDL.mdarray2polyphase(x, df)

    py = multipleAnalysisPolyphaseMat(fb, px)

    y = if outputMode == :normal
        [ MDCDL.vecblocks2array(py.data[p,:], py.nBlocks, tuple(ones(Integer,D)...)) for p in 1:sum(fb.nChannels) ]
    elseif outputMode == :augumented
        polyphase2mdarray(py)
    else
        error("outputMode=$outputMode: This option is not available.")
    end

    return y
end

function multipleAnalysisPolyphaseMat(cc::MDCDL.Cnsolt{D,1,TF}, pvx::PolyphaseVector{TX,D}) where {TF,TX,D}
    const M = prod(cc.decimationFactor)
    const P = cc.nChannels

    x = pvx.data
    tx = cc.matrixF * flipdim(x, 1)

    const V0 = cc.initMatrices[1] * [ eye(Complex{TF},M) ; zeros(Complex{TF},P-M,M) ]
    ux = V0 * tx

    cc.symmetry * extendAtoms(cc, PolyphaseVector(ux,pvx.nBlocks))
end

function extendAtoms(cc::MDCDL.Cnsolt{D,1,TF}, pvx::PolyphaseVector{TX,D}) where {TF,TX,D}
    const rngUpper = (1:fld(cc.nChannels,2),:)
    const rngLower = (fld(cc.nChannels,2)+1:cc.nChannels,:)

    const nShifts = [ fld.(size(pvx,2), pvx.nBlocks)[d] for d in 1:D ]
    for d = 1:D # original order
        pvx = MDCDL.permutedims(pvx)
        x = pvx.data
        for k = 1:cc.polyphaseOrder[d]
            B = MDCDL.getMatrixB(cc.nChannels, cc.paramAngles[d][k])

            x .= B' * x
            x[rngLower...] = circshift(x[rngLower...], (0, nShifts[d]))
            # if k % 2 == 1
            #     px[rngLower...] = circshift(px[rngLower...],(0, nShifts[d]))
            # else
            #     px[rngUpper...] = circshift(px[rngUpper...],(0, -nShifts[d]))
            # end
            x .= B * x

            x[rngUpper...] = cc.propMatrices[d][2*k-1] * x[rngUpper...]
            x[rngLower...] = cc.propMatrices[d][2*k]   * x[rngLower...]
        end
        pvx.data .= x
    end
    return pvx
end

function multipleAnalysisPolyphaseMat(cc::MDCDL.Rnsolt{D,1,TF}, x::PolyphaseVector{TX,D}) where {TF,TX,D}
    const M = prod(cc.decimationFactor)
    const hM = fld(M,2)
    const P = cc.nChannels
    px = copy(x)

    px .= cc.matrixC * flipdim(x.data, 1)

    # const V0 = cc.initMatrices[1] * [ eye(T,M) ; zeros(T,P-M,M) ]
    const W0 = cc.initMatrices[1] * vcat(eye(TF, hM), zeros(TF, P[1] - hM, hM))
    const U0 = cc.initMatrices[2] * vcat(eye(TF, hM), zeros(TF, P[2] - hM, hM))

    ux = PolyphaseVector(vcat(W0 * px[1:hM, :], U0 * px[hM+1:end, :]), px.nBlocks)

    extendAtoms(cc, ux)
end

function extendAtoms(cc::MDCDL.Rnsolt{D,1,TF}, px::PolyphaseVector{TX,D}) where {TF,TX,D}
    const P = cc.nChannels[1]
    # const rngUpper = (1:P,:)
    const rngLower = ((1:P)+P,:)

    const nShifts = [ fld.(size(px,2),px.nBlocks)[d] for d in 1:D ]
    # for d in cc.directionPermutation
    for d = 1:D # original order
        px = MDCDL.permutedims(px)
        for k = 1:cc.polyphaseOrder[d]

            px .= butterfly(px.data, P)
            px[rngLower...] = circshift(px[rngLower...], (0, nShifts[d]))
            # if k % 2 == 1
            #     px[rngLower...] = circshift(px[rngLower...],(0, nShifts[d]))
            # else
            #     px[rngUpper...] = circshift(px[rngUpper...],(0, -nShifts[d]))
            # end
            px .= butterfly(px.data, P)

            px[rngLower...] = cc.propMatrices[d][k]   * px[rngLower...]
        end
    end
    return px
end

function stepAnalysisBank(pfs::MDCDL.ParallelFB{TF,D}, x::Array{TX,D}) where {TF,TX,D}
    const df = pfs.decimationFactor
    const offset = df .- 1

    [  circshift(MDCDL.downsample(MDCDL.mdfilter(x, f), df, offset), 0) for f in pfs.analysisFilters ]
end
