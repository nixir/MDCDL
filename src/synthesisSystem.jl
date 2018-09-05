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

function synthesize(cc::Cnsolt{TF,D,S}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D,S}
    M = prod(cc.decimationFactor)
    P = cc.nChannels

    uy = concatenateAtoms!(cc, PolyphaseVector(cc.symmetry' * pvy.data, pvy.nBlocks); kwargs...)

    py = (cc.initMatrices[1] * Matrix{Complex{TF}}(I,P,M))' * uy.data
    py .= reverse(cc.matrixF, dims=2)' * py

    PolyphaseVector(py, pvy.nBlocks)
end

function concatenateAtoms!(cc::Cnsolt{TF,D,:TypeI}, pvy::PolyphaseVector{TY,D}; border=:circular) where {TF,TY,D}
    P = cc.nChannels

    for d = D:-1:1
        nShift = fld(size(pvy.data,2), pvy.nBlocks[end])
        # submatrices
        y  = view(pvy.data, :, :)
        yu = view(pvy.data, 1:fld(P,2), :)
        yl = view(pvy.data, (fld(P,2)+1):P, :)
        for k = cc.polyphaseOrder[d]:-1:1
            yu .= cc.propMatrices[d][2*k-1]' * yu
            yl .= cc.propMatrices[d][2*k]'   * yl

            B = getMatrixB(P, cc.paramAngles[d][k])
            y .= B' * y

            if isodd(k)
                shiftBackward!(Val{border}, yl, nShift)
            else
                shiftForward!(Val{border}, yu, nShift)
            end
            y .= B * y
        end
        pvy = ipermutedims(pvy)
    end
    return pvy
end


function concatenateAtoms!(cc::Cnsolt{TF,D,:TypeII}, pvy::PolyphaseVector{TY,D}; border=:circular) where {TF,TY,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = cc.nChannels
    chEven = 1:(P-1)

    for d = D:-1:1
        nShift = fld(size(pvy.data,2), pvy.nBlocks[end])
        # submatrices
        ye  = view(pvy.data, 1:(P-1), :)
        yu1 = view(pvy.data, 1:fld(P,2), :)
        yl1 = view(pvy.data, (fld(P,2)+1):(P-1), :)
        yu2 = view(pvy.data, 1:cld(P,2), :)
        yl2 = view(pvy.data, cld(P,2):P, :)
        for k = nStages[d]:-1:1
            # second step

            yu2 .= cc.propMatrices[d][4*k-1]' * yu2
            yl2 .= cc.propMatrices[d][4*k]'   * yl2

            B = getMatrixB(P, cc.paramAngles[d][2*k])
            ye  .= B' * ye
            shiftForward!(Val{border}, yu1, nShift)
            ye  .= B * ye

            # first step

            yu1 .= cc.propMatrices[d][4*k-3]' * yu1
            yl1 .= cc.propMatrices[d][4*k-2]' * yl1

            B = getMatrixB(P, cc.paramAngles[d][2*k-1])
            ye  .= B' * ye
            shiftBackward!(Val{border}, yl1, nShift)
            ye  .= B * ye
        end
        pvy = ipermutedims(pvy)
    end
    return pvy
end

function synthesize(cc::Rnsolt{TF,D,S}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D,S}
    M = prod(cc.decimationFactor)
    cM = cld(M,2)
    fM = fld(M,2)
    nch = cc.nChannels

    cpvy = deepcopy(pvy)
    uy = concatenateAtoms!(cc, cpvy; kwargs...)
    y = uy.data

    W0 = cc.initMatrices[1] * Matrix{TF}(I, nch[1], cM)
    U0 = cc.initMatrices[2] * Matrix{TF}(I, nch[2], fM)
    ty = vcat(W0' * y[1:nch[1],:], U0' * y[(nch[1]+1):end,:])
    ty .= reverse(cc.matrixC, dims=2)' * ty
    PolyphaseVector(ty, uy.nBlocks)
end

function concatenateAtoms!(cc::Rnsolt{TF,D,:TypeI}, pvy::PolyphaseVector{TY,D}; border=:circular) where {TF,TY,D}
    hP = cc.nChannels[1]

    for d = D:-1:1
        nShift = fld(size(pvy.data,2), pvy.nBlocks[end])
        # submatrices
        yu = view(pvy.data, 1:hP, :)
        yl = view(pvy.data, (1:hP) .+ hP, :)
        for k = cc.polyphaseOrder[d]:-1:1
            yl .= cc.propMatrices[d][k]' * yl

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
        pvy = ipermutedims(pvy)
    end
    return pvy
end

function concatenateAtoms!(cc::Rnsolt{TF,D,:TypeII}, pvy::PolyphaseVector{TY,D}; border=:circular) where {TF,TY,D}
    nStages = fld.(cc.polyphaseOrder,2)
    P = sum(cc.nChannels)
    maxP, minP, chMajor, chMinor = if cc.nChannels[1] > cc.nChannels[2]
        (cc.nChannels[1], cc.nChannels[2], 1:cc.nChannels[1], (cc.nChannels[1]+1):P)
    else
        (cc.nChannels[2], cc.nChannels[1], (cc.nChannels[1]+1):P, 1:cc.nChannels[1])
    end

    for d = D:-1:1
        nShift = fld(size(pvy.data,2), pvy.nBlocks[end])
        # submatrices
        yu  = view(pvy.data, 1:minP, :)
        yl  = view(pvy.data, (P-minP+1):P, :)
        ys1 = view(pvy.data, (minP+1):P, :)
        ys2 = view(pvy.data, 1:maxP, :)
        ymj = view(pvy.data, chMajor, :)
        ymn = view(pvy.data, chMinor, :)
        for k = nStages[d]:-1:1
            # second step
            ymj .= cc.propMatrices[d][2*k]' * ymj

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl

            shiftForward!(Val{border}, ys2, nShift)

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl

            # first step
            ymn .= cc.propMatrices[d][2*k-1]' * ymn

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl

            shiftBackward!(Val{border}, ys1, nShift)

            tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
            yu .= tu; yl .= tl
        end
        pvy = ipermutedims(pvy)
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
