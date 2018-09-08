using ImageFiltering: imfilter, reflect, FIR
using OffsetArrays: OffsetArray

function synthesize(syn::NsoltOperator{TF,D}, y::AbstractArray) where {TF,D}
    pvy = if syn.shape == :normal
        nBlocks = size(y[1])
        PolyphaseVector( Matrix(transpose(hcat(map(vec, y)...))), nBlocks)
    elseif syn.shape == :augumented
        mdarray2polyphase(y)
    elseif syn.shape == :vector
        ty = reshape(y, fld.(syn.insize, syn.nsolt.decimationFactor)..., sum(syn.nsolt.nChannels))
        mdarray2polyphase(ty)
    else
        error("Invalid argument.")
    end

    pvx = synthesize(syn.nsolt, pvy; border=syn.border)
    polyphase2mdarray(pvx, syn.nsolt.decimationFactor)
end
operate(::Type{Val{:synthesizer}}, syn::NsoltOperator, y::AbstractArray) = synthesize(syn, y)

function synthesize(cc::Cnsolt{TF,D}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D}
    x = synthesize_cnsolt(Val{cc.category}, pvy.data, pvy.nBlocks, cc.matrixF, cc.initMatrices, cc.propMatrices, cc.paramAngles, cc.symmetry, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels)

    PolyphaseVector(x, pvy.nBlocks)
end

function synthesize_cnsolt(category::Type, y::AbstractMatrix, nBlocks::NTuple{D}, matrixF::AbstractMatrix, initMts::AbstractArray, propMts::AbstractArray, paramAngs::AbstractArray, sym::AbstractMatrix, df::NTuple{D}, ord::NTuple{D}, nch::Integer; kwargs...) where {T,D}
    uy = concatenateAtoms_cnsolt(category, sym' * y, nBlocks, propMts, paramAngs, ord, nch; kwargs...)

    # output = (V0 * F * J)' * uy == J * F' * V0' * uy
    (initMts[1] * Matrix(I, nch, prod(df)) * reverse(matrixF, dims=2))' * uy
end

function concatenateAtoms_cnsolt(::Type{Val{:TypeI}}, pvy::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, paramAngs::AbstractArray, ord::NTuple{D}, P::Integer; border=:circular) where {D}
    pvy = copy(pvy)
    for d = D:-1:1
        nShift = fld(size(pvy, 2), nBlocks[d])
        # submatrices
        y  = view(pvy, :, :)
        yu = view(pvy, 1:fld(P,2), :)
        yl = view(pvy, (fld(P,2)+1):P, :)
        for k = ord[d]:-1:1
            yu .= propMts[d][2*k-1]' * yu
            yl .= propMts[d][2*k]'   * yl

            B = getMatrixB(P, paramAngs[d][k])
            y .= B' * y

            if isodd(k)
                shiftBackward!(Val{border}, yl, nShift)
            else
                shiftForward!(Val{border}, yu, nShift)
            end
            y .= B * y
        end
        pvy = ipermutedimspv(pvy, nBlocks[d])
    end
    return pvy
end

function concatenateAtoms_cnsolt(::Type{Val{:TypeII}}, pvy::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, paramAngs::AbstractArray, ord::NTuple{D}, P::Integer; border=:circular) where {D}
    nStages = fld.(ord, 2)
    chEven = 1:(P-1)

    pvy = copy(pvy)
    for d = D:-1:1
        nShift = fld(size(pvy,2), nBlocks[d])
        # submatrices
        ye  = view(pvy, 1:(P-1), :)
        yu1 = view(pvy, 1:fld(P,2), :)
        yl1 = view(pvy, (fld(P,2)+1):(P-1), :)
        yu2 = view(pvy, 1:cld(P,2), :)
        yl2 = view(pvy, cld(P,2):P, :)
        for k = nStages[d]:-1:1
            # second step
            yu2 .= propMts[d][4*k-1]' * yu2
            yl2 .= propMts[d][4*k]'   * yl2

            B = getMatrixB(P, paramAngs[d][2*k])
            ye  .= B' * ye
            shiftForward!(Val{border}, yu1, nShift)
            ye  .= B * ye

            # first step
            yu1 .= propMts[d][4*k-3]' * yu1
            yl1 .= propMts[d][4*k-2]' * yl1

            B = getMatrixB(P, paramAngs[d][2*k-1])
            ye  .= B' * ye
            shiftBackward!(Val{border}, yl1, nShift)
            ye  .= B * ye
        end
        pvy = ipermutedimspv(pvy, nBlocks[d])
    end
    return pvy
end

function synthesize(cc::Rnsolt{TF,D}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D}
    x = synthesize_rnsolt(Val{cc.category}, pvy.data, pvy.nBlocks, cc.matrixC, cc.initMatrices, cc.propMatrices, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels; kwargs...)
    PolyphaseVector(x, pvy.nBlocks)
end

function synthesize_rnsolt(category::Type, pvy::AbstractMatrix, nBlocks::NTuple{D}, matrixC::AbstractMatrix, initMts::AbstractArray, propMts::AbstractArray, df::NTuple{D}, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {D}
    M = prod(df)

    y = concatenateAtoms_rnsolt(category, pvy, nBlocks, propMts, ord, nch; kwargs...)

    W0 = initMts[1] * Matrix(I, nch[1], cld(M,2))
    U0 = initMts[2] * Matrix(I, nch[2], fld(M,2))
    reverse(matrixC, dims=2)' * vcat(W0' * y[1:nch[1],:], U0' * y[(nch[1]+1):end,:])
end

function concatenateAtoms_rnsolt(::Type{Val{:TypeI}}, pvy::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, ord::NTuple{D}, nch::Tuple{Int,Int}; border=:circular) where {D}
    hP = nch[1]
    pvy = copy(pvy)
    for d = D:-1:1
        nShift = fld(size(pvy, 2), nBlocks[d])
        # submatrices
        yu = view(pvy, 1:hP, :)
        yl = view(pvy, (1:hP) .+ hP, :)
        for k = ord[d]:-1:1
            yl .= propMts[d][k]' * yl

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl
            if isodd(k)
                shiftBackward!(Val{border}, yl, nShift)
            else
                shiftForward!(Val{border}, yu, nShift)
            end
            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl
        end
        pvy = ipermutedimspv(pvy, nBlocks[d])
    end
    return pvy
end

function concatenateAtoms_rnsolt(::Type{Val{:TypeII}}, pvy::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, ord::NTuple{D}, nch::Tuple{Int,Int}; border=:circular) where {D}
    nStages = fld.(ord, 2)
    P = sum(nch)
    maxP, minP, chMajor, chMinor = if nch[1] > nch[2]
        (nch[1], nch[2], 1:nch[1], (nch[1]+1):P)
    else
        (nch[2], nch[1], (nch[1]+1):P, 1:nch[1])
    end

    pvy = copy(pvy)
    for d = D:-1:1
        nShift = fld(size(pvy,2), nBlocks[d])
        # submatrices
        yu  = view(pvy, 1:minP, :)
        yl  = view(pvy, (P-minP+1):P, :)
        ys1 = view(pvy, (minP+1):P, :)
        ys2 = view(pvy, 1:maxP, :)
        ymj = view(pvy, chMajor, :)
        ymn = view(pvy, chMinor, :)
        for k = nStages[d]:-1:1
            # second step
            ymj .= propMts[d][2*k]' * ymj

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl

            shiftForward!(Val{border}, ys2, nShift)

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl

            # first step
            ymn .= propMts[d][2*k-1]' * ymn

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl

            shiftBackward!(Val{border}, ys1, nShift)

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl
        end
        pvy = ipermutedimspv(pvy, nBlocks[d])
    end
    return pvy
end

function synthesize(msop::MultiscaleOperator{TF,D}, y::AbstractArray) where {TF,D}
    if msop.shape == :normal
        subsynthesize(Val{msop.shape}, msop.operators, y)
    elseif msop.shape == :vector
        subsynthesize(Val{msop.shape}, msop.operators, y)
    end
end

function subsynthesize(v::Type{Val{:normal}}, abop::AbstractVector, sy::AbstractArray)
    ya = if length(abop) <= 1
        sy[1]
    else
        [ subsynthesize(v, abop[2:end], sy[2:end]), sy[1]... ]
    end
    synthesize(abop[1], ya)
end

function subsynthesize(v::Type{Val{:vector}}, abop::AbstractVector, sy::AbstractArray)
    ya = if length(abop) <= 1
        sy
    else
        lny = prod(abop[1].outsize) - prod(abop[2].insize)
        [ vec(subsynthesize(v, abop[2:end], sy[lny+1:end])); sy[1:lny] ]
    end
    synthesize(abop[1], ya)
end

operate(::Type{Val{:synthesizer}}, msop::MultiscaleOperator, y::AbstractArray) = synthesize(msop, y)

function synthesize(cs::ConvolutionalOperator{TF,D}, y::AbstractVector) where {TF,D}
    df = cs.decimationFactor
    ord = cs.polyphaseOrder

    alg = if cs.domain == :spacial
        FIR()
    elseif cs.domain == :frequency
        FFT()
    end

    ty = if cs.shape == :normal
        y
    elseif cs.shape == :augumented
        [ y[fill(:,D)..., p] for p in 1:cs.nChannels ]
    elseif cs.shape == :vector
        ry = reshape(y, cs.insize..., cs.nChannels)
        [ ry[fill(:,D)..., p] for p in 1:cs.nChannels ]
    else
        error("Invalid augument")
    end

    nShift = df .* cld.(ord, 2) .+ 1
    region = UnitRange.(1 .- nShift, df .* (ord .+ 1) .- nShift)

    sxs = map(ty, cs.kernels) do yp, sfp
        upimg = upsample(yp, df)
        ker = reflect(OffsetArray(sfp, region...))
        imfilter(upimg, ker, "circular", alg)
    end
    sum(sxs)
end

operate(::Type{Val{:synthesizer}}, cvop::ConvolutionalOperator, y::AbstractArray) = synthesize(cvop, y)
