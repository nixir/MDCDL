using ImageFiltering: imfilter, reflect, FIR
using OffsetArrays: OffsetArray

function synthesize(syn::NsoltOperator{TF,D}, y::AbstractArray) where {TF,D}
    pvy = reshape_coefs(syn.shape, syn, y)

    pvx = synthesize(syn.nsolt, pvy; border=syn.border)
    polyphase2mdarray(pvx, syn.nsolt.decimationFactor)
end

reshape_coefs(::Shapes.Default, ::NsoltOperator, y::AbstractArray) = PolyphaseVector(hcat(vec.(y)...) |> transpose |> Matrix, size(y[1]))
reshape_coefs(::Shapes.Augumented, ::NsoltOperator, y::AbstractArray) = mdarray2polyphase(y)
function reshape_coefs(::Shapes.Vec, nsop::NsoltOperator, y::AbstractArray)
    szout = fld.(nsop.insize, nsop.nsolt.decimationFactor)
    ty = reshape(y, szout..., sum(nsop.nsolt.nChannels))
    mdarray2polyphase(ty)
end

synthesize(fb::PolyphaseFB{TF,D}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D} = PolyphaseVector(synthesize(fb, pvy.data, pvy.nBlocks; kwargs...), pvy.nBlocks)

synthesize(cc::NS, py::AbstractMatrix{TY}, nBlocks::NTuple{D}; kwargs...) where {TF,TY,D,NS<:Cnsolt{TF,D}} = synthesize(NS, Val{cc.category}, py, nBlocks, cc.matrixF, cc.initMatrices, cc.propMatrices, cc.paramAngles, cc.symmetry, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels)

function synthesize(::Type{NS}, ::Type{CT}, y::AbstractMatrix, nBlocks::NTuple{D}, matrixF::AbstractMatrix, initMts::AbstractArray{TM}, propMts::AbstractArray, paramAngs::AbstractArray, sym::AbstractMatrix, df::NTuple{D}, ord::NTuple{D}, nch::Integer; kwargs...) where {TM<:AbstractMatrix,D,TF,NS<:Cnsolt{TF,D},CT}
    uy = concatenateAtoms(NS, CT, sym' * y, nBlocks, propMts, paramAngs, ord, nch; kwargs...)

    # output = (V0 * F * J)' * uy == J * F' * V0' * uy
    (initMts[1] * Matrix(I, nch, prod(df)) * reverse(matrixF, dims=2))' * uy
end

function concatenateAtoms(::Type{NS}, ::Type{CT}, pvy::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, paramAngs::AbstractArray, ord::NTuple{D}, P::Integer; kwargs...) where {TF,D,NS<:Cnsolt{TF,D},CT}
    foldr(1:D; init=pvy) do d, ty
        concatenateAtomsPerDims(NS, CT, ty, nBlocks[d], propMts[d], paramAngs[d], ord[d], P; kwargs...)
    end
end

function concatenateAtomsPerDims(::Type{NS}, ::Type{Val{:TypeI}}, pvy::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TM<:AbstractMatrix,NS<:Cnsolt}
    pvy = TM(Matrix(I,sum(P),sum(P))) * pvy
    nShift = fld(size(pvy, 2), nBlock)
    # submatrices
    y  = view(pvy, :, :)
    yu = view(pvy, 1:fld(P,2), :)
    yl = view(pvy, (fld(P,2)+1):P, :)
    for k = ordd:-1:1
        yu .= propMtsd[2k-1]' * yu
        yl .= propMtsd[2k]'   * yl

        B = getMatrixB(P, paramAngsd[k])
        y .= B' * y

        if isodd(k)
            shiftbackward!(Val{border}, yl, nShift)
        else
            shiftforward!(Val{border}, yu, nShift)
        end
        y .= B * y
    end
    return ipermutedimspv(pvy, nBlock)
end

function concatenateAtomsPerDims(::Type{NS}, ::Type{Val{:TypeII}}, pvy::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TM<:AbstractMatrix,NS<:Cnsolt}
    nStages = fld(ordd, 2)
    chEven = 1:(P-1)

    pvy = TM(Matrix(I,sum(P),sum(P))) * pvy
    nShift = fld(size(pvy,2), nBlock)
    # submatrices
    ye  = view(pvy, 1:(P-1), :)
    yu1 = view(pvy, 1:fld(P,2), :)
    yl1 = view(pvy, (fld(P,2)+1):(P-1), :)
    yu2 = view(pvy, 1:cld(P,2), :)
    yl2 = view(pvy, cld(P,2):P, :)
    for k = nStages:-1:1
        # second step
        yu2 .= propMtsd[4k-1]' * yu2
        yl2 .= propMtsd[4k]'   * yl2

        B = getMatrixB(P, paramAngsd[2k])
        ye  .= B' * ye
        shiftforward!(Val{border}, yu1, nShift)
        ye  .= B * ye

        # first step
        yu1 .= propMtsd[4k-3]' * yu1
        yl1 .= propMtsd[4k-2]' * yl1

        B = getMatrixB(P, paramAngsd[2k-1])
        ye  .= B' * ye
        shiftbackward!(Val{border}, yl1, nShift)
        ye  .= B * ye
    end
    return ipermutedimspv(pvy, nBlock)
end

synthesize(cc::NS, py::AbstractMatrix{TY}, nBlocks::NTuple{D}; kwargs...) where {TF,TY,D,NS<:Rnsolt{TF,D}} = synthesize(NS, Val{cc.category}, py, nBlocks, cc.matrixC, cc.initMatrices, cc.propMatrices, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels; kwargs...)

function synthesize(cc::Type{NS}, ::Type{CT}, pvy::AbstractMatrix, nBlocks::NTuple{D}, matrixC::AbstractMatrix, initMts::AbstractArray{TM}, propMts::AbstractArray, df::NTuple{D}, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {TM<:AbstractMatrix,D,TF,NS<:Rnsolt{TF,D},CT}
    M = prod(df)

    y = concatenateAtoms(NS, CT, pvy, nBlocks, propMts, ord, nch; kwargs...)

    W0 = initMts[1] * Matrix(I, nch[1], cld(M,2))
    U0 = initMts[2] * Matrix(I, nch[2], fld(M,2))
    reverse(matrixC, dims=2)' * vcat(W0' * y[1:nch[1],:], U0' * y[(nch[1]+1):end,:])
end

function concatenateAtoms(::Type{NS}, ::Type{CT}, pvy::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {TF,D,NS<:Rnsolt{TF,D},CT}
    foldr(1:D; init=pvy) do d, ty
        concatenateAtomsPerDims(NS, CT, ty, nBlocks[d], propMts[d], ord[d], nch; kwargs...)
    end
end

function concatenateAtomsPerDims(::Type{NS}, ::Type{Val{:TypeI}}, pvy::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, ordd::Integer, nch::Tuple{Int,Int}; border=:circular) where {TM<:AbstractMatrix,NS<:Rnsolt}
    hP = nch[1]
    pvy = TM(Matrix(I,sum(nch),sum(nch))) * pvy
    nShift = fld(size(pvy, 2), nBlock)
    # submatrices
    yu = view(pvy, 1:hP, :)
    yl = view(pvy, (1:hP) .+ hP, :)
    for k = ordd:-1:1
        yl .= propMtsd[k]' * yl

        tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
        yu .= tu; yl .= tl
        if isodd(k)
            shiftbackward!(Val{border}, yl, nShift)
        else
            shiftforward!(Val{border}, yu, nShift)
        end
        tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
        yu .= tu; yl .= tl
    end
    return ipermutedimspv(pvy, nBlock)
end

function concatenateAtomsPerDims(::Type{NS}, ::Type{Val{:TypeII}}, pvy::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, ordd::Integer, nch::Tuple{Int,Int}; border=:circular) where {TM<:AbstractMatrix,NS<:Rnsolt}
    nStages = fld(ordd, 2)
    P = sum(nch)
    maxP, minP, chMajor, chMinor = if nch[1] > nch[2]
        (nch[1], nch[2], 1:nch[1], (nch[1]+1):P)
    else
        (nch[2], nch[1], (nch[1]+1):P, 1:nch[1])
    end

    pvy = TM(Matrix(I,sum(nch),sum(nch))) * pvy
    nShift = fld(size(pvy,2), nBlock)
    # submatrices
    yu  = view(pvy, 1:minP, :)
    yl  = view(pvy, (P-minP+1):P, :)
    ys1 = view(pvy, (minP+1):P, :)
    ys2 = view(pvy, 1:maxP, :)
    ymj = view(pvy, chMajor, :)
    ymn = view(pvy, chMinor, :)
    for k = nStages:-1:1
        # second step
        ymj .= propMtsd[2k]' * ymj

        tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
        yu .= tu; yl .= tl

        shiftforward!(Val{border}, ys2, nShift)

        tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
        yu .= tu; yl .= tl

        # first step
        ymn .= propMtsd[2k-1]' * ymn

        tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
        yu .= tu; yl .= tl

        shiftbackward!(Val{border}, ys1, nShift)

        tu, tl = (yu + yl, yu - yl) ./ sqrt(2)
        yu .= tu; yl .= tl
    end
    ipermutedimspv(pvy, nBlock)
end

function synthesize(msop::MultiscaleOperator{TF,D}, y::AbstractArray) where {TF,D}
    # if msop.shape isa Shapes.Default
    #     subsynthesize(msop.shape, msop.operators, y)
    # elseif msop.shape isa Shapes.Vec
    #     subsynthesize(msop.shape, msop.operators, y)
    # end
    subsynthesize(msop.shape, msop.operators, y)
end

function subsynthesize(v::Shapes.Default, abop::AbstractVector, sy::AbstractArray)
    ya = if length(abop) <= 1
        sy[1]
    else
        [ subsynthesize(v, abop[2:end], sy[2:end]), sy[1]... ]
    end
    synthesize(abop[1], ya)
end

function subsynthesize(v::Shapes.Vec, abop::AbstractVector, sy::AbstractArray)
    ya = if length(abop) <= 1
        sy
    else
        lny = prod(abop[1].outsize) - prod(abop[2].insize)
        [ vec(subsynthesize(v, abop[2:end], sy[lny+1:end])); sy[1:lny] ]
    end
    synthesize(abop[1], ya)
end

function synthesize(cs::ConvolutionalOperator{TF,D}, y::AbstractVector) where {TF,D}
    df = cs.decimationFactor
    ord = cs.polyphaseOrder

    ty = reshape_coefs(cs.shape, cs, y)

    nShift = df .* cld.(ord, 2) .+ 1
    region = UnitRange.(1 .- nShift, df .* (ord .+ 1) .- nShift)

    sxs = map(ty, cs.kernels) do yp, sfp
        upimg = upsample(yp, df)
        ker = reflect(OffsetArray(sfp, region...))
        imfilter(cs.resource, upimg, ker, "circular")
    end
    sum(sxs)
end

reshape_coefs(::Shapes.Default, ::ConvolutionalOperator, y::AbstractArray) = y
reshape_coefs(::Shapes.Augumented, co::ConvolutionalOperator{T,D}, y::AbstractArray) where {T,D} = [ y[fill(:,D)..., p] for p in 1:co.nChannels]
function reshape_coefs(::Shapes.Vec, co::ConvolutionalOperator{T,D}, y::AbstractArray) where {T,D}
    ry = reshape(y, fld.(co.insize, co.decimationFactor)..., co.nChannels)
    [ ry[fill(:,D)..., p] for p in 1:co.nChannels ]
end
