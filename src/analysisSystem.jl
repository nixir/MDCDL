using ImageFiltering
using OffsetArrays
# Finit-dimensional lienar operator
analyze(mtx::Matrix{T}, x) where T<:Number = mtx * x
adjoint_synthesize(mtx::Matrix{T}, x) where T<:Number = mtx' * x

#
analyze(fb::FilterBank, x; kwargs...) = analyze(fb, x, 1; kwargs...)[1]

# Filter bank with polyphase representation
function analyze(fb::PolyphaseFB{TF,D}, x::Array{TX,D}, args...; kwargs...) where {TF,TX,D}
    analyze(fb, mdarray2polyphase(x, fb.decimationFactor), args...; kwargs...)
end
adjoint_synthesize(fb::PolyphaseFB{TF,D}, x::Array{TX,D}, args...; kwargs...) where {TF,TX,D} = analyze(fb, x, args...; kwargs...)

function analyze(fb::PolyphaseFB{TF,D}, x::PolyphaseVector{TX,D}, level::Integer; outputMode=:reshaped) where {TF,TX,D}
    function subanalyze(sx::PolyphaseVector{TS,D}, k::Integer) where TS
        sy = multipleAnalysisBank(fb, sx)
        if k <= 1
            return [ sy ]
        else
            nondcs = PolyphaseVector(sy.data[2:end,:], sy.nBlocks)

            # reshape Low-pass MD filter
            dcCoefs = PolyphaseVector(sy.data[1:1,:], sy.nBlocks)
            dcData = polyphase2mdarray(dcCoefs, tuple(fill(1,D)...))
            nsx = mdarray2polyphase(dcData, fb.decimationFactor)

            [ nondcs, subanalyze(nsx, k-1)... ]
        end
    end

    y = subanalyze(x,level)

    if outputMode == :polyphase
        y
    elseif outputMode == :reshaped
        map(y) do py
            map(1:size(py.data,1)) do p
                reshape(py.data[p,:], py.nBlocks)
            end
        end
    elseif outputMode == :augumented
        polyphase2mdarray.(y)
    end
end
adjoint_synthesize(fb::PolyphaseFB{TF,D}, x::PolyphaseVector{TX,D}, args...; kwargs...) where {TF,TX,D} = analyze(fb, x, args...; kwargs...)

function multipleAnalysisBank(cc::Cnsolt{TF,D,S}, pvx::PolyphaseVector{TX,D}) where {TF,TX,D,S}
    M = prod(cc.decimationFactor)
    P = cc.nChannels

    x = pvx.data
    tx = cc.matrixF * flipdim(x, 1)

    V0 = cc.initMatrices[1] * [ eye(Complex{TF},M) ; zeros(Complex{TF},P-M,M) ]
    ux = V0 * tx

    extx = extendAtoms(cc, PolyphaseVector(ux,pvx.nBlocks))
    PolyphaseVector(cc.symmetry*extx.data, extx.nBlocks)
end

function extendAtoms(cc::Cnsolt{TF,D,:TypeI}, pvx::PolyphaseVector{TX,D}; boundary=:circular) where {TF,TX,D}
    P = cc.nChannels

    for d = 1:D
        nShift = fld(size(pvx,2), pvx.nBlocks[1])
        pvx = permutedims(pvx)
        x = pvx.data
        # submatrices
        xu = view(x, 1:fld(P, 2), :)
        xl = view(x, (fld(P, 2)+1):P, :)
        for k = 1:cc.polyphaseOrder[d]
            B = getMatrixB(P, cc.paramAngles[d][k])

            x .= B' * x
            if isodd(k)
                xl .= circshift(xl,(0, nShift))
            else
                xu .= circshift(xu,(0, -nShift))
            end
            x .= B * x

            xu .= cc.propMatrices[d][2*k-1] * xu
            xl .= cc.propMatrices[d][2*k]   * xl
        end
        pvx.data .= x
    end
    return pvx
end

function extendAtoms(cc::Cnsolt{TF,D,:TypeII}, pvx::PolyphaseVector{TX,D}; boundary=:circular) where {TF,TX,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = cc.nChannels

    for d = 1:D
        nShift = fld(size(pvx,2), pvx.nBlocks[1])
        pvx = permutedims(pvx)
        x = pvx.data
        # submatrices
        xe  = view(x, 1:P-1, :)
        xu1 = view(x, 1:fld(P,2), :)
        xl1 = view(x, (fld(P,2)+1):(P-1), :)
        xu2 = view(x, 1:cld(P,2), :)
        xl2 = view(x, cld(P,2):P, :)
        for k = 1:nStages[d]
            # first step
            B = getMatrixB(P, cc.paramAngles[d][2*k-1])

            xe  .= B' * xe
            xl1 .= circshift(xl1, (0, nShift))
            xe  .= B * xe

            xu1 .= cc.propMatrices[d][4*k-3] * xu1
            xl1 .= cc.propMatrices[d][4*k-2] * xl1

            # second step
            B = getMatrixB(P, cc.paramAngles[d][2*k])

            xe  .= B' * xe
            xu1 .= circshift(xu1, (0, -nShift))
            xe  .= B * xe

            xl2 .= cc.propMatrices[d][4*k]   * xl2
            xu2 .= cc.propMatrices[d][4*k-1] * xu2
        end
        pvx.data .= x
    end
    return pvx
end

function multipleAnalysisBank(cc::Rnsolt{TF,D,S}, pvx::PolyphaseVector{TX,D}) where {TF,TX,D,S}
    M = prod(cc.decimationFactor)
    cM = cld(M,2)
    fM = fld(M,2)
    P = cc.nChannels
    x = pvx.data

    tx = cc.matrixC * flipdim(x, 1)

    W0 = cc.initMatrices[1] * vcat(eye(TF, cM), zeros(TF, P[1] - cM, cM))
    U0 = cc.initMatrices[2] * vcat(eye(TF, fM), zeros(TF, P[2] - fM, fM))

    ux = PolyphaseVector(vcat(W0 * tx[1:cM, :], U0 * tx[cM+1:end, :]), pvx.nBlocks)

    extendAtoms(cc, ux)
end

function extendAtoms(cc::Rnsolt{TF,D,:TypeI}, pvx::PolyphaseVector{TX,D}, boundary=:circular) where {TF,TX,D}
    hP = cc.nChannels[1]

    for d = 1:D
        nShift = fld(size(pvx,2), pvx.nBlocks[1])
        pvx = permutedims(pvx)
        x = pvx.data
        # submatrices
        xu = view(x, 1:hP, :)
        xl = view(x, (1:hP)+hP, :)
        for k = 1:cc.polyphaseOrder[d]
            x .= butterfly(x, hP)

            if isodd(k)
                xl .= circshift(xl, (0, nShift))
            else
                xu .= circshift(xu, (0, -nShift))
            end
            x .= butterfly(x, hP)

            xl .= cc.propMatrices[d][k] * xl
        end
        pvx.data .= x
    end
    return pvx
end

function extendAtoms(cc::Rnsolt{TF,D,:TypeII}, pvx::PolyphaseVector{TX,D}; boundary=:circular) where {TF,TX,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = sum(cc.nChannels)
    maxP, minP, chMajor, chMinor = if cc.nChannels[1] > cc.nChannels[2]
        (cc.nChannels[1], cc.nChannels[2], 1:cc.nChannels[1], (cc.nChannels[1]+1):P)
    else
        (cc.nChannels[2], cc.nChannels[1], (cc.nChannels[1]+1):P, 1:cc.nChannels[1])
    end

    for d = 1:D
        nShift = fld(size(pvx,2), pvx.nBlocks[1])
        pvx = permutedims(pvx)
        x = pvx.data
        # submatrices
        xs1 = view(x, minP+1:P, :)
        xs2 = view(x, 1:maxP, :)
        xmj = view(x, chMajor, :)
        xmn = view(x, chMinor, :)
        for k = 1:nStages[d]
            # first step
            x   .= butterfly(x, minP)
            xs1 .= circshift(xs1, (0, nShift))
            x   .= butterfly(x, minP)

            xmn .= cc.propMatrices[d][2*k-1] * xmn

            # second step
            x   .= butterfly(x, minP)
            xs2 .= circshift(xs2, (0, -nShift))
            x   .= butterfly(x, minP)

            xmj .= cc.propMatrices[d][2*k] * xmj
        end
        pvx.data .= x
    end
    return pvx
end

function analyze(pfb::ParallelFB{TF,D}, x::Array{TX,D}, level::Integer) where {TF,TX,D}
    df = pfb.decimationFactor
    ord = pfb.polyphaseOrder
    offset = df .- 1
    region = colon.(1,df.*(ord.+1)) .- df.*fld.(ord,2) .- 1

    function subanalyze(sx::Array{TS,D}, k::Integer) where TS
        sy = map(pfb.analysisFilters) do f
            ker = reflect(OffsetArray(f, region...))
            fltimg = imfilter(sx, ker, "circular", ImageFiltering.FIR())
            downsample(fltimg, df, offset)
        end
        if k <= 1
            return [ sy ]
        else
            [ sy[2:end], subanalyze(sy[1], k-1)... ]
        end
    end

    subanalyze(x,level)
end
adjoint_synthesize(pfb::ParallelFB{TF,D}, x::Array{TX,D}, args...; kwargs...) where {TF,TX,D} = analyze(pdf, x, args...; kwargs...)
