
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

function stepAnalysisBank(fb::MDCDL.PolyphaseFB{TF,D}, x::Array{TX,D}; outputMode="normal") where {TF,TX,D}
    const df = fb.decimationFactor
    const M = prod(df)
    const nBlocks = fld.(size(x), df)

    blkx = MDCDL.array2vecblocks(x, df)
    tx = reshape(blkx, M, cld(length(blkx), M))

    ty = multipleAnalysisPolyphaseMat(fb, tx, nBlocks)

    y = if outputMode == "normal"
        [ MDCDL.vecblocks2array(ty[p,:], nBlocks, tuple(ones(Integer,D)...)) for p in 1:sum(fb.nChannels) ]
    elseif outputMode == "augumented"
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
    const S = fld(size(x,2),nBlocks)

    dst = copy(x)
    # for idx = 0:nBlocks-1
    foreach(0:nBlocks-1) do idx
        dst[:, (1:S)+idx*S] = x[:, (1:nBlocks:end)+idx]
    end
    dst
end

function stepAnalysisBank(pfs::MDCDL.ParallelFB{TF,D}, x::Array{TX,D}) where {TF,TX,D}
    const df = pfs.decimationFactor
    const offset = df .- 1

    [  circshift(MDCDL.downsample(MDCDL.mdfilter(x, f), df, offset), 0) for f in pfs.analysisFilters ]
end

################################################################################
### Old systems ###
#
# function analyzeVec(cc::MDCDL.Cnsolt{D,S,T}, x::Array{Complex{T},D}, level::Integer = 1) where {D,S,T}
#     function subanalyzeVec(sx::Array{Complex{T},D}, k::Integer)
#         y = stepAnalysisBankVec(cc,sx)
#         if k <= 1
#             Array{Complex{T},2}[y]
#         else
#             szSubImg = fld.(size(sx), cc.decimationFactor)
#             res = Array{Array{Complex{T},2}}(k)
#             res[1] = y[2:end,:]
#             res[2:end] =  subanalyzeVec(reshape(y[1,:],szSubImg...), k-1)
#             return res
#         end
#     end
#     scales = [ fld.(size(x), cc.decimationFactor.^l) for l in 0:level]
#     (subanalyzeVec(x, level), scales )
# end
#
# function stepAnalysisBankVec(cc::MDCDL.Cnsolt{D,S,T}, x::Array{Complex{T},D}) where {D,S,T}
#     # 実装が面倒なので初期行列は単純なものにする
#     df = cc.decimationFactor
#     M = prod(df)
#     P = sum(cc.nChannels)
#     nBlocks = fld.(size(x), df)
#
#     # vectorize D-dimensional arrays per sub-blocks
#     vblocks = MDCDL.blocks2vec(x, df)
#
#     #
#     px = flipdim(reshape(vblocks, M, cld(length(x),M) ), 1)
#
#     px = cc.matrixF * px
#
#     V0 = cc.initMatrices[1] * [ eye(T,M) ; zeros(T,P-M,M) ]
#     ux = V0 * px
#
#     cc.Symmetry * extendAtoms(cc, ux, nBlocks)
# end
#
# function blocks2vec(x::Array{T,D}, scale::NTuple{D}) where {T,D}
#     nBlocks = fld.(size(x),scale)
#     primeBlock = ntuple(n -> 1:scale[n], D)
#
#     out = Array{Vector{T},D}(nBlocks...)
#     # for cr in CartesianRange(nBlocks)
#     foreach(CartesianRange(nBlocks)) do cr
#         block = (cr.I .- 1) .* scale .+ primeBlock
#         out[cr] = vec(x[block...])
#     end
#     vcat(out...)
# end
