using ImageFiltering: imfilter, reflect, FIR, FFT
using OffsetArrays: OffsetArray

function analyze(A::NsoltOperator{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D}
    y = analyze(A.nsolt, x; border=A.border)
    reshape_polyvec(Val{A.shape}, A, y)
end

reshape_polyvec(::Type{Val{:normal}}, ::NsoltOperator, pvy::PolyphaseVector) = [ reshape(pvy.data[p,:], pvy.nBlocks) for p in 1:size(pvy.data,1) ]
reshape_polyvec(::Type{Val{:augumented}}, ::NsoltOperator, pvy::PolyphaseVector) = polyphase2mdarray(pvy)
reshape_polyvec(::Type{Val{:vector}}, ::NsoltOperator, pvy::PolyphaseVector) = vec(transpose(pvy.data))

analyze(fb::PolyphaseFB{TF,D}, x::AbstractArray{TX,D}, args...; kwargs...) where {TF,TX,D} = analyze(fb, mdarray2polyphase(x, fb.decimationFactor), args...; kwargs...)

analyze(fb::PolyphaseFB{TF,D}, pvx::PolyphaseVector{TX,D}; kwargs...) where {TF,TX,D} = PolyphaseVector(analyze(fb, pvx.data, pvx.nBlocks; kwargs...), pvx.nBlocks)

analyze(cc::NS, px::AbstractMatrix{TX}, nBlocks::NTuple{D}; kwargs...) where {TF,TX,D,NS<:Cnsolt{TF,D}} = analyze(NS, Val{cc.category}, px, nBlocks, cc.matrixF, cc.initMatrices, cc.propMatrices, cc.paramAngles, cc.symmetry, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels; kwargs...)

function analyze(::Type{NS}, ::Type{CT}, x::AbstractMatrix, nBlocks::NTuple{D}, matrixF::AbstractMatrix, initMts::AbstractArray{TM}, propMts::AbstractArray, paramAngs::AbstractArray, sym::AbstractMatrix, df::NTuple{D}, ord::NTuple{D}, nch::Integer; kwargs...) where {TM<:AbstractMatrix,TF,D,NS<:Cnsolt{TF,D},CT}
    # ux = V0 * F * J * x
    ux = (initMts[1] * Matrix(I, nch, prod(df)) * reverse(matrixF, dims=2)) * x

    sym * extendAtoms(NS, CT, ux, nBlocks, propMts, paramAngs, ord, nch; kwargs...)
end

function extendAtoms(::Type{NS}, ::Type{CT}, pvx::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray,  paramAngs::AbstractArray, ord::NTuple{D}, P::Integer; kwargs...) where {D,TF,NS<:Cnsolt{TF,D},CT}
    foldl(1:D; init=pvx) do tx, d
        extendAtomsPerDims(NS, CT, tx, nBlocks[d], propMts[d], paramAngs[d], ord[d], P; kwargs...)
    end
end

function extendAtomsPerDims(::Type{NS}, ::Type{Val{:TypeI}}, pvx::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM},  paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TM<:AbstractMatrix,NS<:Cnsolt}
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
            shiftforward!(Val{border}, xl, nShift)
        else
            shiftbackward!(Val{border}, xu, nShift)
        end
        x .= B * x

        xu .= propMtsd[2k-1] * xu
        xl .= propMtsd[2k]   * xl
    end
    return pvx
end

function extendAtomsPerDims(::Type{NS}, ::Type{Val{:TypeII}}, pvx::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM},  paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TM<:AbstractMatrix,NS<:Cnsolt}
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
        shiftforward!(Val{border}, xl1, nShift)
        xe  .= B * xe

        xu1 .= propMtsd[4k-3] * xu1
        xl1 .= propMtsd[4k-2] * xl1

        # second step
        B = getMatrixB(P, paramAngsd[2k])

        xe  .= B' * xe
        shiftbackward!(Val{border}, xu1, nShift)
        xe  .= B * xe

        xl2 .= propMtsd[4k]   * xl2
        xu2 .= propMtsd[4k-1] * xu2
    end
    return pvx
end

analyze(cc::NS, px::AbstractMatrix{TX}, nBlocks::NTuple{D}; kwargs...) where {TF,TX,D,NS<:Rnsolt{TF,D}} = analyze(NS, Val{cc.category}, px, nBlocks, cc.matrixC, cc.initMatrices, cc.propMatrices, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels; kwargs...)

function analyze(::Type{NS}, ::Type{CT}, x::AbstractMatrix, nBlocks::NTuple, matrixC::AbstractMatrix, initMts::AbstractArray{TM}, propMts::AbstractArray, df::NTuple{D}, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {TM<:AbstractMatrix,D,TF,NS<:Rnsolt{TF,D},CT}
    M = prod(df)
    cM = cld(M,2)
    fM = fld(M,2)

    tx = reverse(matrixC, dims=2) * x

    W0 = initMts[1] * Matrix(I, nch[1], cM)
    U0 = initMts[2] * Matrix(I, nch[2], fM)
    ux = vcat(W0 * tx[1:cM, :], U0 * tx[(cM+1):end, :])

    extendAtoms(NS, CT, ux, nBlocks, propMts, ord, nch; kwargs...)
end

function extendAtoms(::Type{NS}, ::Type{CT}, pvx::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {TF,D,NS<:Rnsolt{TF,D},CT}
    foldl(1:D; init=pvx) do tx, d
        extendAtomsPerDims(NS, CT, tx, nBlocks[d], propMts[d], ord[d], nch; kwargs...)
    end
end

function extendAtomsPerDims(::Type{NS}, ::Type{Val{:TypeI}}, pvx::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, ordd::Integer, nch::Tuple{Int,Int}; border=:circular) where {TM<:AbstractMatrix,NS<:Rnsolt}
    hP = nch[1]
    nShift = fld(size(pvx, 2), nBlock)
    pvx = permutedimspv(pvx, nBlock)
    # submatrices
    xu = view(pvx, 1:hP, :)
    xl = view(pvx, (1:hP) .+ hP, :)
    for k = 1:ordd
        tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
        xu .= tu; xl .= tl

        if isodd(k)
            shiftforward!(Val{border}, xl, nShift)
        else
            shiftbackward!(Val{border}, xu, nShift)
        end
        tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
        xu .= tu; xl .= tl

        xl .= propMtsd[k] * xl
    end
    return pvx
end

function extendAtomsPerDims(::Type{NS}, ::Type{Val{:TypeII}}, pvx::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, ordd::Integer, nch::Tuple{Int,Int}; border=:circular) where {TM<:AbstractMatrix,NS<:Rnsolt}
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
        tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
        xu .= tu; xl .= tl

        shiftforward!(Val{border}, xs1, nShift)

        tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
        xu .= tu; xl .= tl

        xmn .= propMtsd[2k-1] * xmn

        # second step
        tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
        xu .= tu; xl .= tl

        shiftbackward!(Val{border}, xs2, nShift)

        tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
        xu .= tu; xl .= tl

        xmj .= propMtsd[2k] * xmj
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

function analyze(ca::ConvolutionalOperator{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D}
    df = ca.decimationFactor
    ord = ca.polyphaseOrder

    nShift = df .* fld.(ord, 2) .+ 1
    region = UnitRange.(1 .- nShift, df .* (ord .+ 1) .- nShift)

    offset = df .- 1
    y = map(ca.kernels) do f
        ker = reflect(OffsetArray(f, region...))
        fltimg = imfilter(ca.resource, x, ker, "circular")
        downsample(fltimg, df, offset)
    end
    reshape_polyvec(Val{ca.shape}, ca, y)
end

reshape_polyvec(::Type{Val{:normal}}, ::ConvolutionalOperator, y::AbstractArray) = y
reshape_polyvec(::Type{Val{:augumented}}, ::ConvolutionalOperator{TF,D}, y::AbstractArray{TY,D}) where {TF,TY,D} = cat(D+1, y...)
reshape_polyvec(::Type{Val{:vector}}, ::ConvolutionalOperator, y::AbstractArray) = vcat(vec.(y)...)
