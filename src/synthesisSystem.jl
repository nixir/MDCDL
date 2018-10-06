using ImageFiltering: imfilter, reflect, FIR
using OffsetArrays: OffsetArray

function synthesize(syn::TransformSystem{NS}, y::AbstractArray) where {TF,D,NS<:AbstractNsolt{TF,D}}
    y = reshape_coefs(syn.shape, syn.operator, y)
    pvx = synthesize(syn.operator, y; syn.options...)
    polyphase2mdarray(pvx, decimations(syn.operator))
end

synthesize(fb::PolyphaseFB{TF,D}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D} = PolyphaseVector(synthesize(fb, pvy.data, pvy.nBlocks; kwargs...), pvy.nBlocks)

synthesize(cc::NS, py::AbstractMatrix, nBlocks::NTuple{D}; kwargs...) where {TF,D,NS<:Cnsolt{TF,D}} = synthesize(NS, Val(istype1(cc)), py, nBlocks, cc.matrixF, cc.initMatrices, cc.propMatrices, cc.paramAngles, cc.symmetry, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels)

function synthesize(::Type{NS}, tp::Val, y::AbstractMatrix, nBlocks::NTuple, matrixF::AbstractMatrix, initMts::AbstractArray, propMts::AbstractArray, paramAngs::AbstractArray, sym::AbstractMatrix, df::NTuple, ord::NTuple, nch::Integer; kwargs...) where {NS<:Cnsolt}
    y = concatenateAtoms(NS, tp, sym' * y, nBlocks, propMts, paramAngs, ord, nch; kwargs...)
    finalStep(NS, tp, y, matrixF, initMts, df, nch; kwargs...)
end

function finalStep(::Type{NS}, ::Val, y::AbstractMatrix, matrixF::AbstractMatrix, initMts::AbstractArray{TM}, df::NTuple, nch::Integer; kwargs...) where {TM<:AbstractMatrix, NS<:Cnsolt}
    # output = (V0 * F * J)' * uy == J * F' * V0' * uy
    (initMts[1] * Matrix(I, nch, prod(df)) * reverse(matrixF, dims=2))' * y
end

function concatenateAtoms(::Type{NS}, tp::Val, pvy::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, paramAngs::AbstractArray, ord::NTuple{D}, P::Integer; kwargs...) where {TF,D,NS<:Cnsolt{TF,D}}
    foldr(1:D; init=pvy) do d, ty
        concatenateAtomsPerDims(NS, tp, ty, nBlocks[d], propMts[d], paramAngs[d], ord[d], P; kwargs...)
    end
end

function concatenateAtomsPerDims(::Type{NS}, ::TypeI, pvy::AbstractMatrix{TP}, nBlock::Integer, propMtsd::AbstractArray{TM}, paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TN,TP,TM<:AbstractMatrix,NS<:Cnsolt{TN}}
    # pvy = TM(Matrix(I,sum(P),sum(P))) * pvy
    pvy = convert(Array{promote_type(TN,TP)}, pvy)
    nShift = fld(size(pvy, 2), nBlock)
    # submatrices
    y  = @view pvy[:,:]
    yu = @view pvy[1:fld(P,2),:]
    yl = @view pvy[(fld(P,2)+1):P,:]
    for k = ordd:-1:1
        yu .= propMtsd[2k-1]' * yu
        yl .= propMtsd[2k]'   * yl

        B = getMatrixB(P, paramAngsd[k])
        y .= B' * y

        if isodd(k)
            shiftbackward!(Val(border), yl, nShift)
        else
            shiftforward!(Val(border), yu, nShift)
        end
        y .= B * y
    end
    return ishiftdimspv(pvy, nBlock)
end

function concatenateAtomsPerDims(::Type{NS}, ::TypeII, pvy::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TM<:AbstractMatrix,NS<:Cnsolt}
    nStages = fld(ordd, 2)
    chEven = 1:(P-1)

    pvy = TM(Matrix(I,sum(P),sum(P))) * pvy
    nShift = fld(size(pvy,2), nBlock)
    # submatrices
    ye  = @view pvy[1:(P-1),:]
    yu1 = @view pvy[1:fld(P,2),:]
    yl1 = @view pvy[(fld(P,2)+1):(P-1),:]
    yu2 = @view pvy[1:cld(P,2),:]
    yl2 = @view pvy[cld(P,2):P,:]
    for k = nStages:-1:1
        # second step
        yu2 .= propMtsd[4k-1]' * yu2
        yl2 .= propMtsd[4k]'   * yl2

        B = getMatrixB(P, paramAngsd[2k])
        ye  .= B' * ye
        shiftforward!(Val(border), yu1, nShift)
        ye  .= B * ye

        # first step
        yu1 .= propMtsd[4k-3]' * yu1
        yl1 .= propMtsd[4k-2]' * yl1

        B = getMatrixB(P, paramAngsd[2k-1])
        ye  .= B' * ye
        shiftbackward!(Val(border), yl1, nShift)
        ye  .= B * ye
    end
    return ishiftdimspv(pvy, nBlock)
end

synthesize(cc::NS, py::AbstractMatrix, nBlocks::NTuple{D}; kwargs...) where {TF,D,NS<:Rnsolt{TF,D}} = synthesize(NS, Val(istype1(cc)), py, nBlocks, cc.matrixC, cc.initMatrices, cc.propMatrices, cc.decimationFactor, cc.polyphaseOrder, cc.nChannels; kwargs...)

function synthesize(::Type{NS}, tp::Val, pvy::AbstractMatrix, nBlocks::NTuple{D}, matrixC::AbstractMatrix, initMts::AbstractArray{TM}, propMts::AbstractArray, df::NTuple{D}, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {TM<:AbstractMatrix,D,TF,NS<:Rnsolt{TF,D}}
    uy = concatenateAtoms(NS, tp, pvy, nBlocks, propMts, ord, nch; kwargs...)
    finalStep(NS, tp, uy, matrixC, initMts, df, nch; kwargs...)
end

function finalStep(::Type{NS}, ::Val, y::AbstractMatrix, matrixC::AbstractMatrix, initMts::AbstractArray{TM}, df::NTuple, nch::Tuple{Int,Int}; kwargs...) where {TM<:AbstractMatrix,NS<:Rnsolt}
    M = prod(df)
    W0ty = (initMts[1] * Matrix(I, nch[1], cld(M,2)))' * @view y[1:nch[1],:]
    U0ty = (initMts[2] * Matrix(I, nch[2], fld(M,2)))' * @view y[(nch[1]+1):end,:]

    reverse(matrixC, dims=2)' * vcat(W0ty, U0ty)
end

function concatenateAtoms(::Type{NS}, tp::Val, pvy::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, ord::NTuple{D}, nch::Tuple{Int,Int}; kwargs...) where {TF,D,NS<:Rnsolt{TF,D}}
    foldr(1:D; init=pvy) do d, ty
        concatenateAtomsPerDims(NS, tp, ty, nBlocks[d], propMts[d], ord[d], nch; kwargs...)
    end
end

function concatenateAtomsPerDims(::Type{NS}, ::TypeI, pvy::AbstractMatrix{TP}, nBlock::Integer, propMtsd::AbstractArray{TM}, ordd::Integer, nch::Tuple{Int,Int}; border=:circular) where {TP,TN,TM<:AbstractMatrix,NS<:Rnsolt{TN}}
    hP = nch[1]
    # pvy = TM(Matrix(I,sum(nch),sum(nch))) * pvy
    pvy = convert(Array{promote_type(TN,TP)}, pvy)
    nShift = fld(size(pvy, 2), nBlock)
    # submatrices
    yu = @view pvy[1:hP,:]
    yl = @view pvy[(1:hP) .+ hP,:]
    for k = ordd:-1:1
        yl .= propMtsd[k]' * yl

        unnormalized_butterfly!(yu, yl)
        if isodd(k)
            shiftbackward!(Val(border), yl, nShift)
        else
            shiftforward!(Val(border), yu, nShift)
        end
        half_butterfly!(yu, yl)
    end
    return ishiftdimspv(pvy, nBlock)
end

function concatenateAtomsPerDims(::Type{NS}, ::TypeII, pvy::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, ordd::Integer, nch::Tuple{Int,Int}; border=:circular) where {TM<:AbstractMatrix,NS<:Rnsolt}
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
    yu  = @view pvy[1:minP,:]
    yl  = @view pvy[(P-minP+1):P,:]
    ys1 = @view pvy[(minP+1):P,:]
    ys2 = @view pvy[1:maxP,:]
    ymj = @view pvy[chMajor,:]
    ymn = @view pvy[chMinor,:]
    for k = nStages:-1:1
        # second step
        ymj .= propMtsd[2k]' * ymj

        unnormalized_butterfly!(yu, yl)
        shiftforward!(Val(border), ys2, nShift)
        half_butterfly!(yu, yl)

        # first step
        ymn .= propMtsd[2k-1]' * ymn

        unnormalized_butterfly!(yu, yl)
        shiftbackward!(Val(border), ys1, nShift)
        half_butterfly!(yu, yl)
    end
    ishiftdimspv(pvy, nBlock)
end

function synthesize(jts::JoinedTransformSystems{MS}, y::AbstractArray) where {MS<:Multiscale}
    subsynthesize(jts.shape, y, jts.transforms...)
end

# shape == Shapes.Separated
function subsynthesize(shape::Shapes.Separated, sy::AbstractArray, abop::AbstractOperator, args...)
    ya = [ subsynthesize(shape, sy[2:end], args...), sy[1]... ]
    synthesize(abop, ya)
end

subsynthesize(::Shapes.Separated, sy::AbstractArray, abop::AbstractOperator) = synthesize(abop, sy[1])

# shape == Shapes.Combined
function subsynthesize(shape::Shapes.Combined, sy::AbstractArray, abop::AbstractOperator, args...)
    rx = subsynthesize(shape, sy[2:end], args...)
    synthesize(abop, cat(rx, sy[1]; dims=ndims(sy[1]) ))
end

subsynthesize(::Shapes.Combined, sy::AbstractArray, abop::AbstractOperator) = synthesize(abop, sy[1])

# shape == Shapes.Vec
function subsynthesize(shape::Shapes.Vec, sy::AbstractArray, abop::AbstractOperator, args...)
    lny = nchannels(abop) * prod(fld.(abop.shape.insize, decimations(abop))) - prod(args[1].shape.insize)

    ya = [ vec(subsynthesize(shape, sy[lny+1:end], args...)); sy[1:lny] ]
    synthesize(abop, ya)
end

subsynthesize(::Shapes.Vec, sy::AbstractArray, abop::AbstractOperator) = synthesize(abop, sy)

function synthesize(pfs::ParallelFilters{TF,D}, y::AbstractArray; resource=CPU1(FIR())) where {TF,D}
    df = pfs.decimationFactor
    ord = pfs.polyphaseOrder

    nShift = df .* cld.(ord, 2) .+ 1
    region = UnitRange.(1 .- nShift, df .* (ord .+ 1) .- nShift)

    sxs = map(y, pfs.kernelspair[2]) do yp, sfp
        upimg = upsample(yp, df)
        ker = reflect(OffsetArray(sfp, region...))
        imfilter(resource, upimg, ker, "circular")
    end
    sum(sxs)
end

function synthesize(cs::TransformSystem{PF}, y::AbstractVector) where {TF,D,PF<:ParallelFilters{TF,D}}
    ty = reshape_coefs(cs.shape, cs.operator, y)
    synthesize(cs.operator, ty; cs.options...)
end
