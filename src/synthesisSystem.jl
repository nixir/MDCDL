
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

function stepSynthesisBank(fb::MDCDL.PolyphaseFB{TF,D}, y::Array{TY}, szRegion::NTuple{D}; inputMode=:normal) where {TF,TY,D}
    const M = prod(fb.decimationFactor)
    const P = sum(fb.nChannels)
    const nBlocks = fld.(szRegion, fb.decimationFactor)

    ty = if inputMode == :normal
        vcat([MDCDL.array2vecblocks(y[p], nBlocks).' for p in 1:P]...)
    elseif inputMode == :augumented
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

function multipleSynthesisPolyphaseMat(cc::MDCDL.Cnsolt{D,1,TF}, y::Matrix{TY}, nBlocks::NTuple{D}) where {TF,TY,D}
    const M = prod(cc.decimationFactor)
    const P = cc.nChannels

    uy = concatenateAtoms(cc, cc.symmetry' * y, nBlocks)

    py = [ eye(Complex{TF},M) ; zeros(Complex{TF},P-M,M) ]' * cc.initMatrices[1]' *  uy

    py = cc.matrixF' * py

    flipdim(py, 1)
end

# function multipleAnalysisPolyphaseMat(cc::MDCDL.Cnsolt, x::Matrix{T}, nBlocks) where T<:Real
#     multipleSynthesisPolyphaseMat(cc, complex(x), nBlocks)
# end

function concatenateAtoms(cc::MDCDL.Cnsolt{D,1,TF}, py::Array{TY,2}, nBlocks::NTuple{D}) where {TF,TY,D}
    const rngUpper = (1:fld(cc.nChannels,2),:)
    const rngLower = (fld(cc.nChannels,2)+1:cc.nChannels,:)

    const nShifts = [ -1 .* fld.(size(py,2),nBlocks)[d] for d in 1:D ]

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

function multipleSynthesisPolyphaseMat(cc::MDCDL.Rnsolt{D,1,TF}, y::Matrix{TY}, nBlocks::NTuple{D}) where {TF,TY,D}
    const M = prod(cc.decimationFactor)
    const hM = fld(M,2)
    const P = cc.nChannels

    uy = concatenateAtoms(cc, y, nBlocks)

    const W0 = cc.initMatrices[1] * vcat(eye(TF, hM), zeros(TF, P[1] - hM, hM))
    const U0 = cc.initMatrices[2] * vcat(eye(TF, hM), zeros(TF, P[2] - hM, hM))
    py = vcat(W0' * uy[1:P[1],:], U0' * uy[P[1]+1:end,:])
    # py = [ eye(Complex{TF},M) ; zeros(Complex{TF},P-M,M) ]' * cc.initMatrices[1]' *  uy

    py = cc.matrixC' * py

    flipdim(py, 1)
end

function concatenateAtoms(cc::MDCDL.Rnsolt{D,1,TF}, py::Array{TY,2}, nBlocks::NTuple{D}) where {TF,TY,D}
    const P = cc.nChannels[1]
    # const rngUpper = (1:P,:)
    const rngLower = ((1:P)+P,:)

    const nShifts = [ -1 .* fld.(size(py,2),nBlocks)[d] for d in 1:D ]

    for d = D:-1:1 # original order
        for k = cc.polyphaseOrder[d]:-1:1
            py[rngLower...] = cc.propMatrices[d][k]'   * py[rngLower...]

            py = butterfly(py, P)
            py[rngLower...] = circshift(py[rngLower...], (0, nShifts[d]))
            # if k % 2 == 1
            #     py[rngLower...] = circshift(py[rngLower...],(0, nShifts[d]))
            # else
            #     py[rngUpper...] = circshift(py[rngUpper...],(0, nShifts[d]))
            # end
            py = butterfly(py, P)
        end
        py = ipermutePolyphaseSignalDims(py, nBlocks[d])
    end
    return py
end

function ipermutePolyphaseSignalDims(x::Matrix, nBlocks::Integer)
    const S = fld(size(x,2),nBlocks)
    hcat( [ x[:, (1:S:end) + idx] for idx in 0:S-1]... )
end

function stepSynthesisBank(pfb::MDCDL.ParallelFB{TF,D}, y::Vector{Array{TY,D}}, szRegion) where {TF,TY,D}
    const df = pfb.decimationFactor
    const offset = df .* 0
    sxs = map(y, pfb.synthesisFilters) do yp, sfp
        MDCDL.mdfilter(MDCDL.upsample(yp, df, offset), sfp)
    end
    circshift(sum(sxs), -1 .* df .* pfb.polyphaseOrder)
end
