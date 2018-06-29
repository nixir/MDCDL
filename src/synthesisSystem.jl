
function synthesize(fb::MDCDL.FilterBank{T,D}, y::Array{Array{Array{TY,D},1},1}, scales::Array, level::Integer = 1) where {T,TY,D}
    function subsynthesize(sy::Array{Array{Array{TY,D},1},1}, k::Integer)
        ya = if k <= 1
            sy[1]
        else
            vcat(subsynthesize(sy[2:end],k-1),sy[1])
        end
        stepSynthesisBank(fb, ya, scales[level-k+1])
    end
    vx = subsynthesize(y, level)
    reshape(vx, scales[1]...)
end

function stepSynthesisBank(fb::MDCDL.PolyphaseFB{TF,D}, y::Array{TY}, szRegion::NTuple{D}; inputMode="normal") where {TF,TY,D}
    const M = prod(fb.decimationFactor)
    const P = sum(fb.nChannels)
    const nBlocks = fld.(szRegion, fb.decimationFactor)

    ty = if inputMode == "normal"
        vcat([MDCDL.array2vecblocks(y[p], nBlocks).' for p in 1:P]...)
    elseif inputMode == "augumented"
        const szAugBlock = tuple(nBlocks..., sum(fb.nChannels))

        reshape(MDCDL.array2vecblocks(y, szAugBlock),fld(length(y),P),P).'
    else
        error("inputMode=$inputMode: This option is not available.")
    end

    tx = multipleSynthesisPolyphaseMat(fb, ty, nBlocks)

    blkx = reshape(tx, length(tx))
    vecblocks2array(blkx, szRegion, fb.decimationFactor)

end

function stepSynthesisBank(fb::MDCDL.PolyphaseFB{TF,DF}, y::Array{TY,DY}; kwargs...) where {TF,DF,TY,DY}
    if DY != DF + 1
        error("Dimension mismatch. Dim. of the filter bank = $DF, Dim. of the input = $DY.")
    end
    szRegion = size(y)[1:DY-1] .* fb.decimationFactor
    stepSynthesisBank(fb,y,szRegion; kwargs...)
end

function multipleSynthesisPolyphaseMat(cc::MDCDL.Cnsolt{D,1,T}, y::Matrix{Complex{T}}, nBlocks::NTuple{D}) where {T,D}
    const M = prod(cc.decimationFactor)
    const P = cc.nChannels
    # nBlocks = fld.(szRegion,cc.decimationFactor)

    uy = concatenateAtoms(cc, cc.symmetry' * y, nBlocks)

    py = [ eye(T,M) ; zeros(T,P-M,M) ]' * cc.initMatrices[1]' *  uy

    py = cc.matrixF' * py

    flipdim(py, 1)
end

# function multipleAnalysisPolyphaseMat(cc::MDCDL.Cnsolt, x::Matrix{T}, nBlocks) where T<:Real
#     multipleSynthesisPolyphaseMat(cc, complex(x), nBlocks)
# end

function concatenateAtoms(cc::MDCDL.Cnsolt{D,1,T}, py::Array{Complex{T},2}, nBlocks::NTuple{D}) where {T,D}
    const rngUpper = (1:fld(cc.nChannels,2),:)
    const rngLower = (fld(cc.nChannels,2)+1:cc.nChannels,:)

    const nShifts = [ -1 .* fld.(size(py,2),nBlocks)[d] for d in 1:D ]

    #for d in reverse(cc.directionOrder)
    for d = D:-1:1 # original order
        for k = cc.polyphaseOrder[d]:-1:1
            py[rngUpper...] = cc.propMatrices[d][2*k-1]' * py[rngUpper...]
            py[rngLower...] = cc.propMatrices[d][2*k]'   * py[rngLower...]

            B = MDCDL.getMatrixB(cc.nChannels, cc.paramAngles[d][k])
            py = B' * py
            py[rngLower...] = circshift(py[rngLower...], (0, nShifts[d]))
            # if k % 2 == 1
            #     py[rngLower...] = circshift(py[rngLower...],(0, nShifts[d]))
            # else
            #     py[rngUpper...] = circshift(py[rngUpper...],(0, nShifts[d]))
            # end
            py = B * py
        end
        py = ipermutePolyphaseSignalDims(py, nBlocks[d])
    end
    return py
end

function ipermutePolyphaseSignalDims(x::Matrix, nBlocks::Integer)
    const S = fld(size(x,2),nBlocks)

    dst = copy(x)
    for idx = 0:nBlocks-1
        dst[:, (1:nBlocks:end)+idx] = x[:, (1:S)+idx*S]
    end
    dst
end

function stepSynthesisBank(pfb::MDCDL.ParallelFB{TF,D}, y::Vector{Array{TY,D}}, szRegion) where {TF,TY,D}
    const df = pfb.decimationFactor
    const offset = df .* 0
    sxs = map(y, pfb.synthesisFilters) do yp, sfp
        MDCDL.mdfilter(MDCDL.upsample(yp, df, offset), sfp)
    end
    circshift(sum(sxs), -1 .* df .* pfb.polyphaseOrder)
end

###########################################################
#
# function synthesizeVec(cc::MDCDL.Cnsolt{D,S,T}, y::Array{Array{Complex{T},2},1}, scales::Array, level::Integer = 1) where {D,S,T}
#     function subsynthesizeVec(sy::Array{Array{Complex{T},2},1}, k::Integer)
#         ya = if k <= 1
#             sy[1]
#         else
#             vcat(subsynthesizeVec(sy[2:end],k-1).',sy[1])
#         end
#         stepSynthesisBankVec(cc, ya, scales[level-k+1])
#     end
#     vx = subsynthesizeVec(y, level)
#     reshape(vx, scales[1]...)
# end
#
# function stepSynthesisBankVec(cc::MDCDL.Cnsolt{D,S,T}, y::Array{Complex{T}}, szRegion) where {D,S,T}
#     M = prod(cc.decimationFactor)
#     P = cc.nChannels
#
#     nBlocks = fld.(szRegion,cc.decimationFactor)
#
#     uy = concatenateAtoms(cc, cc.Symmetry' * y, nBlocks)
#
#     py = [ eye(T,M) ; zeros(T,P-M,M) ]' * cc.initMatrices[1]' *  uy
#
#     # py = MDCDL.getMatrixOmega(cc)' * py
#     py = cc.matrixF' * py
#
#     vpy = reshape(flipdim(py, 1), length(py))
#
#     # block-wise vectorized operation
#     vec(vec2blocks(vpy,szRegion,cc.decimationFactor))
# end
#
# function vec2blocks(v::Vector{T}, szOut::NTuple{D}, scale::NTuple{D}) where {T,D}
#     len = prod(scale)
#     nBlocks = fld.(szOut,scale)
#
#     tmp = reshape([ reshape(v[(1:len) + s ],scale...) for s in 0:len:length(v)-1 ], nBlocks...)
#
#     primeBlock = ntuple(n -> 1:scale[n], D)
#     out = Array{T,D}(szOut...)
#     for cr in CartesianRange(nBlocks)
#         block = (cr.I .- 1) .* scale .+ primeBlock
#         out[block...] = tmp[cr]
#     end
#     out
# end
