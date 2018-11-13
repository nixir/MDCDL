using ImageFiltering: imfilter, reflect, FIR
using OffsetArrays: OffsetArray

function synthesize(syn::TransformSystem{NS}, y::AbstractArray) where {TF,D,NS<:AbstractNsolt{TF,D}}
    y = reshape_coefs(syn.shape, syn.operator, y)
    pvx = synthesize(syn.operator, y; syn.options...)
    polyphase2mdarray(pvx, decimations(syn.operator))
end

synthesize(fb::PolyphaseFB{TF,D}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D} = PolyphaseVector(synthesize(fb, pvy.data, pvy.nBlocks; kwargs...), pvy.nBlocks)

function synthesize(nsolt::AbstractNsolt, py::AbstractMatrix, nBlocks::NTuple; kwargs...) where {TF,TX,D}
    nShifts = fld.(size(py, 2), nBlocks)
    irotatedimsfcns = ([ t->ishiftdimspv(t, blk) for blk in nBlocks ]...,)
    ty = concatenateAtoms(nsolt, py, nShifts, irotatedimsfcns; kwargs...)

    finalStep(nsolt, ty; kwargs...)
end

function finalStep(nsolt::RnsoltTypeI, py::AbstractMatrix; kwargs...)
    M = prod(nsolt.decimationFactor)
    fM, cM = fld(M, 2), cld(M, 2)

    tyup = nsolt.W0' * py[1:nsolt.nChannels[1], :]
    tylw = nsolt.U0' * py[(1:nsolt.nChannels[2]) .+ nsolt.nChannels[1], :]

    return nsolt.CJ' * [ tyup[1:cM, :]; tylw[1:fM, :] ]
end

function finalStep(nsolt::RnsoltTypeII, py::AbstractMatrix; kwargs...)
    M = prod(nsolt.decimationFactor)
    fM, cM = fld(M, 2), cld(M, 2)

    tyup = nsolt.W0' * py[1:nsolt.nChannels[1], :]
    tylw = nsolt.U0' * py[(1:nsolt.nChannels[2]) .+ nsolt.nChannels[1], :]

    return nsolt.CJ' * [ tyup[1:cM, :]; tylw[1:fM, :] ]
end

# function concatenateAtoms(::Type{NS}, tp::Val, pvy::AbstractMatrix, nBlocks::NTuple{D}, propMts::AbstractArray, paramAngs::AbstractArray, ord::NTuple{D}, P::Integer; kwargs...) where {TF,D,NS<:Cnsolt{TF,D}}
#     foldr(1:D; init=pvy) do d, ty
#         concatenateAtomsPerDims(NS, tp, ty, nBlocks[d], propMts[d], paramAngs[d], ord[d], P; kwargs...)
#     end
# end

function concatenateAtoms(nsolt::RnsoltTypeI, px::AbstractMatrix, nShifts::NTuple, irotatedimsfcns::NTuple; border=:circular)
    px = Array(px)
    params = (irotatedimsfcns, nShifts, nsolt.nStages, nsolt.Udks)
    foreach(reverse.(params)...) do rdfcn, nshift, nstage, Uks
        xu = @view px[1:nsolt.nChannels[1], :]
        xl = @view px[(1:nsolt.nChannels[2]) .+ nsolt.nChannels[1], :]

        params_d = (1:nstage, Uks)
        foreach(reverse.(params_d)...) do k, U
            xl .= U' * xl
            unnormalized_butterfly!(xu, xl)
            if isodd(k)
                shiftbackward!(Val(border), xl, nshift)
            else
                shiftforward!(Val(border), xl, nshift)
            end
            half_butterfly!(xu, xl)
        end
        px = rdfcn(px)
    end
    return px
end

function concatenateAtoms(nsolt::RnsoltTypeII, px::AbstractMatrix, nShifts::NTuple, rotatedimsfcns; border=:circular)
    px = Array(px)
    mnP, mxP = minmax(nsolt.nChannels...)
    params = (rotatedimsfcns, nShifts, nsolt.nStages, nsolt.Wdks, nsolt.Udks)
    foreach(reverse.(params)...) do rdfcn, nshift, nstage, Wks, Uks
        xu = @view px[1:mnP, :]
        xl = @view px[end-mnP+1:end, :]
        xum = @view px[1:mxP, :]
        xml = @view px[end-mxP+1:end, :]

        params_d = (1:nstage, Wks, Uks)
        foreach(reverse.(params_d)...) do k, W, U
            if nsolt.nChannels[1] < nsolt.nChannels[2]
                xml .= U' * xml
            else
                xum .= W' * xum
            end
            unnormalized_butterfly!(xu, xl)
            shiftforward!(Val(border), xl, nshift)
            half_butterfly!(xu, xl)
            if nsolt.nChannels[1] < nsolt.nChannels[2]
                xu .= W' * xu
            else
                xl .= U' * xl
            end
            unnormalized_butterfly!(xu, xl)
            shiftbackward!(Val(border), xml, nshift)
            half_butterfly!(xu, xl)
        end
        px = rdfcn(px)
    end
    return px
end

function finalStep(nsolt::CnsoltTypeI, px::AbstractMatrix; kwargs...)
    nsolt.FJ' * (nsolt.V0' * px)[1:prod(nsolt.decimationFactor), :]
end

function finalStep(nsolt::CnsoltTypeII, px::AbstractMatrix; kwargs...)
    nsolt.FJ' * (nsolt.V0' * px)[1:prod(nsolt.decimationFactor), :]
end

function concatenateAtoms(nsolt::CnsoltTypeI, px::AbstractMatrix, nShifts::NTuple, rotatedimsfcns::NTuple; border=:circular)
    px = Array(px)
    hP = fld(nsolt.nChannels, 2)
    params = (rotatedimsfcns, nShifts, nsolt.nStages, nsolt.Wdks, nsolt.Udks, nsolt.θdks)
    foreach(reverse.(params)...) do rdfcn, nshift, nstage, Wks, Uks, θks
        xu = @view px[1:hP, :]
        xl = @view px[(1:hP) .+ hP, :]

        params_d = (1:nstage, Wks, Uks, θks)
        foreach(reverse.(params_d)...) do k, W, U, θ
            xl .= U' * xl

            B = getMatrixB(nsolt.nChannels, θ)
            px .= B' * px
            if isodd(k)
                shiftbackward!(Val(border), xl, nshift)
            else
                shiftforward!(Val(border), xl, nshift)
            end
            px .= B * px
        end
        px = rdfcn(px)
    end
    return px
end

function concatenateAtoms(nsolt::CnsoltTypeII, px::AbstractMatrix, nShifts::NTuple, rotatedimsfcns; border=:circular)
    px = Array(px)
    fP, cP = fld(nsolt.nChannels, 2), cld(nsolt.nChannels, 2)
    params = (rotatedimsfcns, nShifts, nsolt.nStages, nsolt.Wdks, nsolt.Udks, nsolt.θ1dks, nsolt.Ŵdks, nsolt.Ûdks, nsolt.θ2dks)
    foreach(reverse.(params)...) do rdfcn, nshift, nstage, Wks, Uks, θ1ks, Ŵks, Ûks, θ2ks
        xu1 = @view px[1:fP, :]
        xl1 = @view px[(1:fP) .+ fP, :]
        xu2 = @view px[1:cP, :]
        xl2 = @view px[(1:cP) .+ fP, :]
        xe  = @view px[1:end-1, :]

        params_d = (1:nstage, Wks, Uks, θ1ks, Ŵks, Ûks, θ2ks)
        foreach(reverse.(params_d)...) do k, W, U, θ1, Ŵ, Û, θ2
            xu2 .= Ŵ' * xu2
            xl2 .= Û' * xl2
            B2 = getMatrixB(nsolt.nChannels, θ2)
            xe .= B2' * xe
            shiftforward!(Val(border), xl2, nshift)
            xe .= B2 * xe

            xu1 .= W' * xu1
            xl1 .= U' * xl1
            B1 = getMatrixB(nsolt.nChannels, θ1)
            xe .= B1' * xe
            shiftbackward!(Val(border), xl1, nshift)
            xe .= B1 * xe
        end
        px = rdfcn(px)
    end
    return px
end


# function concatenateAtomsPerDims(::Type{NS}, ::TypeI, pvy::AbstractMatrix{TP}, nBlock::Integer, propMtsd::AbstractArray{TM}, paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TN,TP,TM<:AbstractMatrix,NS<:Cnsolt{TN}}
#     pvy = TM(Matrix(I,sum(P),sum(P))) * pvy
#     # pvy = convert(Array{promote_type(TN),TP)}, pvy)
#     nShift = fld(size(pvy, 2), nBlock)
#     # submatrices
#     y  = @view pvy[:,:]
#     yu = @view pvy[1:fld(P,2),:]
#     yl = @view pvy[(fld(P,2)+1):P,:]
#     for k = ordd:-1:1
#         yu .= propMtsd[2k-1]' * yu
#         yl .= propMtsd[2k]'   * yl
#
#         B = getMatrixB(P, paramAngsd[k])
#         y .= B' * y
#
#         if isodd(k)
#             shiftbackward!(Val(border), yl, nShift)
#         else
#             shiftforward!(Val(border), yu, nShift)
#         end
#         y .= B * y
#     end
#     return ishiftdimspv(pvy, nBlock)
# end
#
# function concatenateAtomsPerDims(::Type{NS}, ::TypeII, pvy::AbstractMatrix, nBlock::Integer, propMtsd::AbstractArray{TM}, paramAngsd::AbstractArray, ordd::Integer, P::Integer; border=:circular) where {TM<:AbstractMatrix,NS<:Cnsolt}
#     nStages = fld(ordd, 2)
#     chEven = 1:(P-1)
#
#     pvy = TM(Matrix(I,sum(P),sum(P))) * pvy
#     nShift = fld(size(pvy,2), nBlock)
#     # submatrices
#     ye  = @view pvy[1:(P-1),:]
#     yu1 = @view pvy[1:fld(P,2),:]
#     yl1 = @view pvy[(fld(P,2)+1):(P-1),:]
#     yu2 = @view pvy[1:cld(P,2),:]
#     yl2 = @view pvy[cld(P,2):P,:]
#     for k = nStages:-1:1
#         # second step
#         yu2 .= propMtsd[4k-1]' * yu2
#         yl2 .= propMtsd[4k]'   * yl2
#
#         B = getMatrixB(P, paramAngsd[2k])
#         ye  .= B' * ye
#         shiftforward!(Val(border), yu1, nShift)
#         ye  .= B * ye
#
#         # first step
#         yu1 .= propMtsd[4k-3]' * yu1
#         yl1 .= propMtsd[4k-2]' * yl1
#
#         B = getMatrixB(P, paramAngsd[2k-1])
#         ye  .= B' * ye
#         shiftbackward!(Val(border), yl1, nShift)
#         ye  .= B * ye
#     end
#     return ishiftdimspv(pvy, nBlock)
# end

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
