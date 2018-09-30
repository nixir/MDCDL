using ImageFiltering: imfilter, reflect, FIR, FFT
using OffsetArrays: OffsetArray

function analyze(A::NsoltOperator{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D}
    pvx = mdarray2polyphase(x, decimations(A.nsolt))
    y = analyze(A.nsolt, pvx; A.options...)
    reshape_polyvec(A.shape, A, y)
end

analyze(fb::PolyphaseFB{TF,D}, pvx::PolyphaseVector{TX,D}; kwargs...) where {TF,TX,D} = PolyphaseVector(analyze(fb, pvx.data, pvx.nBlocks; kwargs...), pvx.nBlocks)

analyze(cc::NS, px::AbstractMatrix, nBlocks::NTuple{D}; kwargs...) where {TF,D,NS<:Cnsolt{TF,D}} = analyze(NS, Val(istype1(cc)), px, nBlocks, cc.matrixF, cc.initMatrices, cc.propMatrices, cc.paramAngles, cc.symmetry, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels; kwargs...)

function analyze(::Type{NS}, tp::Val, x::AbstractMatrix, nBlocks::NTuple, matrixF::AbstractMatrix, initMts::AbstractArray, propMts::AbstractArray, paramAngs::AbstractArray, sym::AbstractMatrix, df::NTuple, ord::NTuple, nch::Integer; kwargs...) where {NS<:Cnsolt}
    ux = initialStep(NS, tp, x, matrixF, initMts, df, nch; kwargs...)
    sym * extendAtoms(NS, tp, ux, nBlocks, propMts, paramAngs, ord, nch; kwargs...)
end

function initialStep(::Type{NS}, ::Val, x::AbstractMatrix, matrixF::AbstractMatrix, initMts::AbstractArray{TM}, df::NTuple, nch::Integer; kwargs...) where {TM<:AbstractMatrix,NS<:Cnsolt}
    # ux = V0 * F * J * x
    (initMts[1] * Matrix(I, nch, prod(df)) * reverse(matrixF, dims=2)) * x
end

function extendAtoms(::Type{NS}, tp::Val, pvx::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray,  paramAngs::AbstractArray, ord::NTuple{D}, P::Integer; kwargs...) where {D,TF,NS<:Cnsolt{TF,D}}
    foldl(1:D; init=pvx) do tx, d
        extendAtomsPerDims(NS, tp, tx, nBlocks[d], propMts[d], paramAngs[d], ord[d], P; kwargs...)
    end
end

function extendAtomsPerDims(::Type{NS}, ::TypeI, pvx::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM},  paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TM<:AbstractMatrix,NS<:Cnsolt}
    nShift = fld(size(pvx, 2), nBlock)
    pvx = permutedimspv(pvx, nBlock)
    # submatrices
    x  = view(pvx, :, :)
    xu = view(pvx, 1:fld(P, 2), :)
    xl = view(pvx, (fld(P, 2)+1):P, :)
    for k = 1:ordd
        B = getMatrixB(P, paramAngsd[k])

        x .= B' * x
        if isodd(k)
            shiftforward!(Val(border), xl, nShift)
        else
            shiftbackward!(Val(border), xu, nShift)
        end
        x .= B * x

        xu .= propMtsd[2k-1] * xu
        xl .= propMtsd[2k]   * xl
    end
    return pvx
end

function extendAtomsPerDims(::Type{NS}, ::TypeII, pvx::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM},  paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TM<:AbstractMatrix,NS<:Cnsolt}
    nStages = fld(ordd, 2)
    nShift = fld(size(pvx, 2), nBlock)
    pvx = permutedimspv(pvx, nBlock)
    # submatrices
    xe  = view(pvx, 1:P-1, :)
    xu1 = view(pvx, 1:fld(P,2), :)
    xl1 = view(pvx, (fld(P,2)+1):(P-1), :)
    xu2 = view(pvx, 1:cld(P,2), :)
    xl2 = view(pvx, cld(P,2):P, :)
    for k = 1:nStages
        # first step
        B = getMatrixB(P, paramAngsd[2k-1])

        xe  .= B' * xe
        shiftforward!(Val(border), xl1, nShift)
        xe  .= B * xe

        xu1 .= propMtsd[4k-3] * xu1
        xl1 .= propMtsd[4k-2] * xl1

        # second step
        B = getMatrixB(P, paramAngsd[2k])

        xe  .= B' * xe
        shiftbackward!(Val(border), xu1, nShift)
        xe  .= B * xe

        xl2 .= propMtsd[4k]   * xl2
        xu2 .= propMtsd[4k-1] * xu2
    end
    return pvx
end

analyze(cc::NS, px::AbstractMatrix, nBlocks::NTuple{D}; kwargs...) where {TF,D,NS<:Rnsolt{TF,D}} = analyze(NS, Val(istype1(cc)), px, nBlocks, cc.matrixC, cc.initMatrices, cc.propMatrices, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels; kwargs...)

function analyze(::Type{NS}, tp::Val, x::AbstractMatrix, nBlocks::NTuple, matrixC::AbstractMatrix, initMts::AbstractArray{TM}, propMts::AbstractArray, df::NTuple{D}, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {TM<:AbstractMatrix,D,TF,NS<:Rnsolt{TF,D}}
    ux = initialStep(NS, tp, x, matrixC, initMts, df, nch; kwargs...)
    extendAtoms(NS, tp, ux, nBlocks, propMts, ord, nch; kwargs...)
end

function initialStep(::Type{NS}, ::Val, x::AbstractMatrix, matrixC::AbstractMatrix, initMts::AbstractArray{TM}, df::NTuple, nch::Tuple{Int,Int}; kwargs...) where {TM<:AbstractMatrix,NS<:Rnsolt}
    M = prod(df)
    cM, fM = cld(M,2), fld(M,2)

    tx = reverse(matrixC, dims=2) * x

    W0x = initMts[1] * Matrix(I, nch[1], cM) * tx[1:cM, :]
    U0x = initMts[2] * Matrix(I, nch[2], fM) * tx[(cM+1):end, :]
    vcat(W0x, U0x)
end

function extendAtoms(::Type{NS}, tp::Val, pvx::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {TF,D,NS<:Rnsolt{TF,D}}
    foldl(1:D; init=pvx) do tx, d
        extendAtomsPerDims(NS, tp, tx, nBlocks[d], propMts[d], ord[d], nch; kwargs...)
    end
end

function extendAtomsPerDims(::Type{NS}, ::TypeI, pvx::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, ordd::Integer, nch::Tuple{Int,Int}; border=:circular) where {TM<:AbstractMatrix,NS<:Rnsolt}
    hP = nch[1]
    nShift = fld(size(pvx, 2), nBlock)
    pvx = permutedimspv(pvx, nBlock)
    # submatrices
    xu = view(pvx, 1:hP, :)
    xl = view(pvx, (1:hP) .+ hP, :)
    for k = 1:ordd
        # pvx .= B * pvx
        unnormalized_butterfly!(xu, xl)
        if isodd(k)
            shiftforward!(Val(border), xl, nShift)
        else
            shiftbackward!(Val(border), xu, nShift)
        end
        # pvx .= 1/2 * B * pvx
        half_butterfly!(xu, xl)

        xl .= propMtsd[k] * xl
    end
    return pvx
end

function extendAtomsPerDims(::Type{NS}, ::TypeII, pvx::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, ordd::Integer, nch::Tuple{Int,Int}; border=:circular) where {TM<:AbstractMatrix,NS<:Rnsolt}
    nStages = fld(ordd, 2)
    P = sum(nch)
    maxP, minP, chMajor, chMinor = if nch[1] > nch[2]
        (nch[1], nch[2], 1:nch[1], (nch[1]+1):P)
    else
        (nch[2], nch[1], (nch[1]+1):P, 1:nch[1])
    end

    nShift = fld(size(pvx,2), nBlock)
    pvx = permutedimspv(pvx, nBlock)
    # submatrices
    xu  = view(pvx, 1:minP, :)
    xl  = view(pvx, (P-minP+1):P, :)
    xs1 = view(pvx, (minP+1):P, :)
    xs2 = view(pvx, 1:maxP, :)
    xmj = view(pvx, chMajor, :)
    xmn = view(pvx, chMinor, :)
    for k = 1:nStages
        # first step
        unnormalized_butterfly!(xu, xl)
        shiftforward!(Val(border), xs1, nShift)
        half_butterfly!(xu, xl)

        xmn .= propMtsd[2k-1] * xmn

        # second step
        unnormalized_butterfly!(xu, xl)
        shiftbackward!(Val(border), xs2, nShift)
        half_butterfly!(xu, xl)

        xmj .= propMtsd[2k] * xmj
    end
    return pvx
end

analyze(msop::MultiscaleOperator{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D} = subanalyze(msop.shape, x, msop.operators...)

# shape == Shapes.Default
function subanalyze(shape::Shapes.Default, sx::AbstractArray, abop::AbstractOperator, args...)
    sy = analyze(abop, sx)
    [sy[2:end], subanalyze(shape, sy[1], args...)...]
end

subanalyze(::Shapes.Default, sx::AbstractArray, abop::AbstractOperator) = [ analyze(abop, sx) ]

# shape == Shapes.Arrayed
function subanalyze(shape::Shapes.Arrayed, sx::AbstractArray{T,D}, abop::AbstractOperator, args...) where {T,D}
    sy = analyze(abop, sx)
    clns = fill(:,D)
    [ sy[clns...,2:end], subanalyze(shape, sy[clns...,1], args...)... ]
end

subanalyze(::Shapes.Arrayed, sx::AbstractArray, abop::AbstractOperator) = [ analyze(abop, sx) ]

# shape == Shapes.Vec
function subanalyze(shape::Shapes.Vec, sx::AbstractArray, abop::AbstractOperator, args...)
    sy = analyze(abop, sx)
    lndc = fld(length(sy), nchannels(abop))
    dcdata = reshape(sy[1:lndc], args[1].shape.insize...)
    vcat(sy[lndc+1:end], subanalyze(shape, dcdata, args...))
end

subanalyze(::Shapes.Vec, sx::AbstractArray, abop::AbstractOperator) = analyze(abop, sx)

function analyze(pfs::ParallelFilters{TF,D}, x::AbstractArray{TX,D}; resource=CPU1(FIR())) where {TF,TX,D}
    df = pfs.decimationFactor
    ord = pfs.polyphaseOrder

    nShift = df .* fld.(ord, 2) .+ 1
    region = UnitRange.(1 .- nShift, df .* (ord .+ 1) .- nShift)
    offset = df .- 1

    map(pfs.kernels) do f
        ker = reflect(OffsetArray(f, region...))
        fltimg = imfilter(resource, x, ker, "circular")
        downsample(fltimg, df, offset)
    end
end

function analyze(ca::ConvolutionalOperator{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D}
    y = analyze(ca.parallelFilters, x; ca.options...)
    reshape_polyvec(ca.shape, ca, y)
end
