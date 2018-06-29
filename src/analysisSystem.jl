
function analyze(fb::MDCDL.FilterBank{TF,D}, x::Array{TX,D}, level::Integer = 1) where {TF,TX,D}
    function subanalyze(sx::Array{TX,D}, k::Integer)
        sy = stepAnalysisBank(fb, sx)
        if k <= 1
            return [ sy ]
        else
            res = Array{Array{Array{TX,D},1},1}(k)
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
    const nBlocks = fld.(size(x), df)

    blkx = MDCDL.array2vecblocks(x, df)
    tx = reshape(blkx, M, cld(length(blkx), M))

    ty = multipleAnalysisPolyphaseMat(fb, tx, nBlocks)

    y = if outputMode == :normal
        [ MDCDL.vecblocks2array(ty[p,:], nBlocks, tuple(ones(Integer,D)...)) for p in 1:sum(fb.nChannels) ]
    elseif outputMode == :augumented
        szAugOut = tuple(nBlocks..., sum(fb.nChannels))
        szAugBlock = tuple(ones(Integer,D+1)...)

        MDCDL.vecblocks2array(vec(ty.'),szAugOut,szAugBlock)
    else
        error("outputMode=$outputMode: This option is not available.")
    end

    return y
end

function multipleAnalysisPolyphaseMat(cc::MDCDL.Cnsolt{D,1,T}, x::Matrix{Complex{T}}, nBlocks::NTuple{D}) where {T,D}
    const M = prod(cc.decimationFactor)
    const P = cc.nChannels

    px = cc.matrixF * flipdim(x, 1)

    const V0 = cc.initMatrices[1] * [ eye(T,M) ; zeros(T,P-M,M) ]
    ux = V0 * px

    cc.symmetry * extendAtoms(cc, ux, nBlocks)
end

# function multipleAnalysisPolyphaseMat(cc::MDCDL.Cnsolt, x::Matrix{T}, nBlocks) where T<:Real
#     multipleAnalysisPolyphaseMat(cc, complex(x), nBlocks)
# end

function extendAtoms(cc::MDCDL.Cnsolt{D,1,T}, px::Matrix{Complex{T}}, nBlocks::NTuple{D}) where {T,D}
    const rngUpper = (1:fld(cc.nChannels,2),:)
    const rngLower = (fld(cc.nChannels,2)+1:cc.nChannels,:)

    const nShifts = [ fld.(size(px,2),nBlocks)[d] for d in 1:D ]
    # for d in cc.directionPermutation
    for d = 1:D # original order
        px = permutePolyphaseSignalDims(px, nBlocks[d])
        for k = 1:cc.polyphaseOrder[d]
            B = MDCDL.getMatrixB(cc.nChannels, cc.paramAngles[d][k])

            px = B' * px
            px[rngLower...] = circshift(px[rngLower...], (0, nShifts[d]))
            # if k % 2 == 1
            #     px[rngLower...] = circshift(px[rngLower...],(0, nShifts[d]))
            # else
            #     px[rngUpper...] = circshift(px[rngUpper...],(0, -nShifts[d]))
            # end
            px = B * px

            px[rngUpper...] = cc.propMatrices[d][2*k-1] * px[rngUpper...]
            px[rngLower...] = cc.propMatrices[d][2*k]   * px[rngLower...]
        end
    end
    return px
end

function permutePolyphaseSignalDims(x::Matrix, nBlocks::Integer)
    hcat( [ x[:, (1:nBlocks:end) + idx] for idx in 0:nBlocks-1 ]... )
end

function stepAnalysisBank(pfs::MDCDL.ParallelFB{TF,D}, x::Array{TX,D}) where {TF,TX,D}
    const df = pfs.decimationFactor
    const offset = df .- 1

    [  circshift(MDCDL.downsample(MDCDL.mdfilter(x, f), df, offset), 0) for f in pfs.analysisFilters ]
end
