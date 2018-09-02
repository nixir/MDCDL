using ImageFiltering: imfilter, reflect, FIR
using OffsetArrays: OffsetArray

function analyze(A::PolyphaseFBAnalyzer{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D}
    y = analyze(A.codebook, x)

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
(ana::PolyphaseFBAnalyzer)(x::AbstractArray) = analyze(ana, x)

function analyze(fb::PolyphaseFB{TF,D}, x::AbstractArray{TX,D}, args...; kwargs...) where {TF,TX,D}
    analyze(fb, mdarray2polyphase(x, fb.decimationFactor), args...; kwargs...)
end

function analyze(cc::Cnsolt{TF,D,S}, pvx::PolyphaseVector{TX,D}; kwargs...) where {TF,TX,D,S}
    M = prod(cc.decimationFactor)
    P = cc.nChannels

    x = pvx.data
    tx = reverse(cc.matrixF, dims=2) * x

    V0 = cc.initMatrices[1] * Matrix{Complex{TF}}(I, P, M)
    ux = V0 * tx

    extx = extendAtoms!(cc, PolyphaseVector(ux, pvx.nBlocks); kwargs...)
    PolyphaseVector(cc.symmetry * extx.data, extx.nBlocks)
end

function extendAtoms!(cc::Cnsolt{TF,D,:TypeI}, pvx::PolyphaseVector{TX,D}; border=:circular) where {TF,TX,D}
    P = cc.nChannels

    for d = 1:D
        nShift = fld(size(pvx.data, 2), pvx.nBlocks[1])
        pvx = permutedims(pvx)
        # submatrices
        x  = view(pvx.data, :, :)
        xu = view(pvx.data, 1:fld(P, 2), :)
        xl = view(pvx.data, (fld(P, 2)+1):P, :)
        for k = 1:cc.polyphaseOrder[d]
            B = getMatrixB(P, cc.paramAngles[d][k])

            x .= B' * x
            if isodd(k)
                shiftForward!(Val{border}, xl, nShift)
            else
                shiftBackward!(Val{border}, xu, nShift)
            end
            x .= B * x

            xu .= cc.propMatrices[d][2*k-1] * xu
            xl .= cc.propMatrices[d][2*k]   * xl
        end
    end
    return pvx
end

function extendAtoms!(cc::Cnsolt{TF,D,:TypeII}, pvx::PolyphaseVector{TX,D}; border=:circular) where {TF,TX,D}
    nStages = fld.(cc.polyphaseOrder, 2)
    P = cc.nChannels

    for d = 1:D
        nShift = fld(size(pvx.data, 2), pvx.nBlocks[1])
        pvx = permutedims(pvx)
        # submatrices
        xe  = view(pvx.data, 1:P-1, :)
        xu1 = view(pvx.data, 1:fld(P,2), :)
        xl1 = view(pvx.data, (fld(P,2)+1):(P-1), :)
        xu2 = view(pvx.data, 1:cld(P,2), :)
        xl2 = view(pvx.data, cld(P,2):P, :)
        for k = 1:nStages[d]
            # first step
            B = getMatrixB(P, cc.paramAngles[d][2*k-1])

            xe  .= B' * xe
            shiftForward!(Val{border}, xl1, nShift)
            xe  .= B * xe

            xu1 .= cc.propMatrices[d][4*k-3] * xu1
            xl1 .= cc.propMatrices[d][4*k-2] * xl1

            # second step
            B = getMatrixB(P, cc.paramAngles[d][2*k])

            xe  .= B' * xe
            shiftBackward!(Val{border}, xu1, nShift)
            xe  .= B * xe

            xl2 .= cc.propMatrices[d][4*k]   * xl2
            xu2 .= cc.propMatrices[d][4*k-1] * xu2
        end
    end
    return pvx
end

function analyze(cc::Rnsolt{TF,D,S}, pvx::PolyphaseVector{TX,D}; kwargs...) where {TF,TX,D,S}
    M = prod(cc.decimationFactor)
    cM = cld(M,2)
    fM = fld(M,2)
    nch = cc.nChannels

    tx = reverse(cc.matrixC, dims=2) * pvx.data

    W0 = cc.initMatrices[1] * Matrix{TF}(I, nch[1], cM)
    U0 = cc.initMatrices[2] * Matrix{TF}(I, nch[2], fM)
    ux = PolyphaseVector(vcat(W0 * tx[1:cM, :], U0 * tx[(cM+1):end, :]), pvx.nBlocks)

    extendAtoms!(cc, ux; kwargs...)
end

function extendAtoms!(cc::Rnsolt{TF,D,:TypeI}, pvx::PolyphaseVector{TX,D}; border=:circular) where {TF,TX,D}
    hP = cc.nChannels[1]

    for d = 1:D
        nShift = fld(size(pvx.data, 2), pvx.nBlocks[1])
        pvx = permutedims(pvx)
        # submatrices
        xu = view(pvx.data, 1:hP, :)
        xl = view(pvx.data, (1:hP) .+ hP, :)
        for k = 1:cc.polyphaseOrder[d]
            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            if isodd(k)
                shiftForward!(Val{border}, xl, nShift)
            else
                shiftBackward!(Val{border}, xu, nShift)
            end
            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            xl .= cc.propMatrices[d][k] * xl
        end
    end
    return pvx
end

function extendAtoms!(cc::Rnsolt{TF,D,:TypeII}, pvx::PolyphaseVector{TX,D}; border=:circular) where {TF,TX,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = sum(cc.nChannels)
    maxP, minP, chMajor, chMinor = if cc.nChannels[1] > cc.nChannels[2]
        (cc.nChannels[1], cc.nChannels[2], 1:cc.nChannels[1], (cc.nChannels[1]+1):P)
    else
        (cc.nChannels[2], cc.nChannels[1], (cc.nChannels[1]+1):P, 1:cc.nChannels[1])
    end

    for d = 1:D
        nShift = fld(size(pvx.data,2), pvx.nBlocks[1])
        pvx = permutedims(pvx)
        # submatrices
        xu  = view(pvx.data, 1:minP, :)
        xl  = view(pvx.data, (P-minP+1):P, :)
        xs1 = view(pvx.data, (minP+1):P, :)
        xs2 = view(pvx.data, 1:maxP, :)
        xmj = view(pvx.data, chMajor, :)
        xmn = view(pvx.data, chMinor, :)
        for k = 1:nStages[d]
            # first step
            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            shiftForward!(Val{border}, xs1, nShift)

            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            xmn .= cc.propMatrices[d][2*k-1] * xmn

            # second step
            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            shiftBackward!(Val{border}, xs2, nShift)

            tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
            xu .= tu; xl .= tl

            xmj .= cc.propMatrices[d][2*k] * xmj
        end
    end
    return pvx
end

function analyze(pfb::ParallelFB{TF,D}, x::AbstractArray{TX,D}; shape=:normal, alg=FIR()) where {TF,TX,D}
    df = pfb.decimationFactor
    ord = pfb.polyphaseOrder

    nShift = df .* fld.(ord, 2) .+ 1
    region = UnitRange.(1 .- nShift, df .* (ord .+ 1) .- nShift)

    offset = df .- 1
    y = map(pfb.analysisFilters) do f
        ker = reflect(OffsetArray(f, region...))
        fltimg = imfilter(x, ker, "circular", alg)
        downsample(fltimg, df, offset)
    end

    if shape == :normal
        y
    elseif shape == :augumented
        cat(D+1, y...)
    elseif shape == :vector
        vcat(vec.(y)...)
    else
        error("Invalid augument")
    end
end
# adjoint_synthesize(pfb::ParallelFB{TF,D}, x::AbstractArray{TX,D}, args...; kwargs...) where {TF,TX,D} = analyze(pfb, x, args...; kwargs...)

function analyze(msfb::Multiscale{TF,D}, x::AbstractArray{TX,D}; shape=:normal) where {TF,TX,D}
    y = subanalyze(msfb.filterBank, x, msfb.treeLevel)
    if shape == :normal
        y
    elseif shape == :augumented
        map(y) do sy
            cat(D+1, sy...)
        end
    elseif shape == :vector
        vty = map(y) do sy
            vcat(vec.(sy)...)
        end
        vcat(vty...)
    end
end
adjoint_synthesize(msfb::Multiscale{TF,D}, x::AbstractArray{TX,D}, args...; kwargs...) where {TF,TX,D} = analyze(msfb, x, args...; kwargs...)

function subanalyze(fb::FilterBank{TF,D}, sx::AbstractArray{TS,D}, k::Integer; kwargs...) where {TF,TS,D}
    sy = analyze(fb, sx; shape=:normal)
    if k <= 1
        [sy]
    else
        [sy[2:end], subanalyze(fb, sy[1], k-1)...]
    end
end
