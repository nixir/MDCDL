using ImageFiltering: imfilter, reflect, FIR, FFT
using OffsetArrays: OffsetArray

function analyze(A::NsoltOperator{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D}
    y = analyze(A.nsolt, x; border=A.border)

    if A.shape == :normal
        [ reshape(y.data[p,:], y.nBlocks) for p in 1:size(y.data,1) ]
    elseif A.shape == :augumented
        polyphase2mdarray(y)
    elseif A.shape == :vector
        vec(transpose(y.data))
    else
        error("Invalid augument.")
    end
end
operate(::Type{Val{:analyzer}}, nsop::NsoltOperator, x::AbstractArray) = analyze(nsop, x)

function analyze(fb::PolyphaseFB{TF,D}, x::AbstractArray{TX,D}, args...; kwargs...) where {TF,TX,D}
    analyze(fb, mdarray2polyphase(x, fb.decimationFactor), args...; kwargs...)
end

function analyze(cc::Cnsolt{TF,D}, pvx::PolyphaseVector{TX,D}; kwargs...) where {TF,TX,D}
    tmp = analyze_cnsolt(Val{cc.category}, pvx.data, pvx.nBlocks, cc.matrixF, cc.initMatrices, cc.propMatrices, cc.paramAngles, cc.symmetry, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels; kwargs...)
    PolyphaseVector(tmp, pvx.nBlocks)
end

function analyze_cnsolt(category::Type, x::AbstractMatrix, nBlocks::NTuple{D}, matrixF::AbstractMatrix, initMts::AbstractArray, propMts::AbstractArray, paramAngs::AbstractArray, sym::AbstractMatrix, df::NTuple{D}, ord::NTuple{D}, nch::Integer; kwargs...) where {T,D}
    # ux = V0 * F * J * x
    ux = (initMts[1] * Matrix(I, nch, prod(df)) * reverse(matrixF, dims=2)) * x

    sym * extendAtoms_cnsolt(category, ux, nBlocks, propMts, paramAngs, ord, nch; kwargs...)
end

function extendAtoms_cnsolt(::Type{Val{:TypeI}}, pvx::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray,  paramAngs::AbstractArray, ord::NTuple{D}, P::Integer; border=:circular) where {D}
    for d = 1:D
        nShift = fld(size(pvx, 2), nBlocks[d])
        pvx = permutedimspv(pvx, nBlocks[d])
        # submatrices
        x  = view(pvx, :, :)
        xu = view(pvx, 1:fld(P, 2), :)
        xl = view(pvx, (fld(P, 2)+1):P, :)
        for k = 1:ord[d]
            B = getMatrixB(P, paramAngs[d][k])

            x .= B' * x
            if isodd(k)
                shiftForward!(Val{border}, xl, nShift)
            else
                shiftBackward!(Val{border}, xu, nShift)
            end
            x .= B * x

            xu .= propMts[d][2*k-1] * xu
            xl .= propMts[d][2*k]   * xl
        end
    end
    return pvx
end

function extendAtoms_cnsolt(::Type{Val{:TypeII}}, pvx::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray,  paramAngs::AbstractArray, ord::NTuple{D}, P::Integer; border=:circular) where {D}
    nStages = fld.(ord, 2)

    for d = 1:D
        nShift = fld(size(pvx, 2), nBlocks[d])
        pvx = permutedimspv(pvx, nBlocks[d])
        # submatrices
        xe  = view(pvx, 1:P-1, :)
        xu1 = view(pvx, 1:fld(P,2), :)
        xl1 = view(pvx, (fld(P,2)+1):(P-1), :)
        xu2 = view(pvx, 1:cld(P,2), :)
        xl2 = view(pvx, cld(P,2):P, :)
        for k = 1:nStages[d]
            # first step
            B = getMatrixB(P, paramAngs[d][2*k-1])

            xe  .= B' * xe
            shiftForward!(Val{border}, xl1, nShift)
            xe  .= B * xe

            xu1 .= propMts[d][4*k-3] * xu1
            xl1 .= propMts[d][4*k-2] * xl1

            # second step
            B = getMatrixB(P, paramAngs[d][2*k])

            xe  .= B' * xe
            shiftBackward!(Val{border}, xu1, nShift)
            xe  .= B * xe

            xl2 .= propMts[d][4*k]   * xl2
            xu2 .= propMts[d][4*k-1] * xu2
        end
    end
    return pvx
end

function analyze(cc::Rnsolt{TF,D}, pvx::PolyphaseVector{TX,D}; kwargs...) where {TF,TX,D}
    tmp = analyze_rnsolt(Val{cc.category}, pvx.data, pvx.nBlocks, cc.matrixC, cc.initMatrices, cc.propMatrices, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels; kwargs...)
    PolyphaseVector(tmp, pvx.nBlocks)
end

function analyze_rnsolt(category::Type, x::AbstractMatrix, nBlocks::NTuple, matrixC::AbstractMatrix, initMts::AbstractArray, propMts::AbstractArray, df::NTuple{D}, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {D}
    M = prod(df)
    cM = cld(M,2)
    fM = fld(M,2)

    tx = reverse(matrixC, dims=2) * x

    W0 = initMts[1] * Matrix(I, nch[1], cM)
    U0 = initMts[2] * Matrix(I, nch[2], fM)
    ux = vcat(W0 * tx[1:cM, :], U0 * tx[(cM+1):end, :])

    extendAtoms_rnsolt(category, ux, nBlocks, propMts, ord, nch; kwargs...)
end

function extendAtoms_rnsolt(::Type{Val{:TypeI}}, pvx::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, ord::NTuple{D}, nch::Tuple{Int,Int}; border=:circular) where {D}
    hP = nch[1]

    for d = 1:D
        nShift = fld(size(pvx, 2), nBlocks[d])
        pvx = permutedimspv(pvx, nBlocks[d])
        # submatrices
        xu = view(pvx, 1:hP, :)
        xl = view(pvx, (1:hP) .+ hP, :)
        for k = 1:ord[d]
            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            if isodd(k)
                shiftForward!(Val{border}, xl, nShift)
            else
                shiftBackward!(Val{border}, xu, nShift)
            end
            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            xl .= propMts[d][k] * xl
        end
    end
    return pvx
end

function extendAtoms_rnsolt(::Type{Val{:TypeII}}, pvx::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, ord::NTuple{D}, nch::Tuple{Int,Int}; border=:circular) where {D}
    nStages = fld.(ord, 2)
    P = sum(nch)
    maxP, minP, chMajor, chMinor = if nch[1] > nch[2]
        (nch[1], nch[2], 1:nch[1], (nch[1]+1):P)
    else
        (nch[2], nch[1], (nch[1]+1):P, 1:nch[1])
    end

    for d = 1:D
        nShift = fld(size(pvx,2), nBlocks[d])
        pvx = permutedimspv(pvx, nBlocks[d])
        # submatrices
        xu  = view(pvx, 1:minP, :)
        xl  = view(pvx, (P-minP+1):P, :)
        xs1 = view(pvx, (minP+1):P, :)
        xs2 = view(pvx, 1:maxP, :)
        xmj = view(pvx, chMajor, :)
        xmn = view(pvx, chMinor, :)
        for k = 1:nStages[d]
            # first step
            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            shiftForward!(Val{border}, xs1, nShift)

            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            xmn .= propMts[d][2*k-1] * xmn

            # second step
            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            shiftBackward!(Val{border}, xs2, nShift)

            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            xmj .= propMts[d][2*k] * xmj
        end
    end
    return pvx
end

function analyze(msop::MultiscaleOperator{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D}
    y = subanalyze(msop.operators, x)
    if msop.shape == :normal
        y
    elseif msop.shape == :augumented
        map(y) do sy
            cat(D+1, sy...)
        end
    elseif msop.shape == :vector
        vty = map(y) do sy
            vcat(vec.(sy)...)
        end
        vcat(vty...)
    end
end

function subanalyze(abop::AbstractVector, sx::AbstractArray{TS,D}) where {TS,D}
    sy = analyze(abop[1], sx)
    if length(abop) <= 1
        [sy]
    else
        [sy[2:end], subanalyze(abop[2:end], sy[1])...]
    end
end

operate(::Type{Val{:analyzer}}, msop::MultiscaleOperator, x::AbstractArray) = analyze(msop, x)

function analyze(ca::ConvolutionalOperator{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D}
    df = ca.decimationFactor
    ord = ca.polyphaseOrder

    alg = if ca.domain == :spacial
        FIR()
    elseif ca.domain == :frequency
        FFT()
    end

    nShift = df .* fld.(ord, 2) .+ 1
    region = UnitRange.(1 .- nShift, df .* (ord .+ 1) .- nShift)

    offset = df .- 1
    y = map(ca.kernels) do f
        ker = reflect(OffsetArray(f, region...))
        fltimg = imfilter(x, ker, "circular", alg)
        downsample(fltimg, df, offset)
    end

    if ca.shape == :normal
        y
    elseif ca.shape == :augumented
        cat(D+1, y...)
    elseif ca.shape == :vector
        vcat(vec.(y)...)
    else
        error("Invalid augument")
    end
end

operate(::Type{Val{:analyzer}}, cvop::ConvolutionalOperator, x::AbstractArray) = analyze(cvop, x)
