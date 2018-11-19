using ImageFiltering: imfilter, reflect, FIR
using OffsetArrays: OffsetArray

function synthesize(syn::TransformSystem{NS}, y::AbstractArray) where {TF,D,NS<:AbstractNsolt{TF,D}}
    y = reshape_coefs(syn.shape, syn.operator, y)
    pvx = synthesize(syn.operator, y; syn.options...)
    polyphase2mdarray(pvx, decimations(syn.operator))
end

synthesize(fb::PolyphaseFB{TF,D}, pvy::PolyphaseVector{TY,D}; kwargs...) where {TF,TY,D} = PolyphaseVector(synthesize(fb, pvy.data, pvy.nBlocks; kwargs...), pvy.nBlocks)

function synthesize(nsolt::AbstractNsolt, py::AbstractMatrix, nBlocks::NTuple; kwargs...) where {TF,TX,D}
    sy = ishiftFilterSymmetry(nsolt, py)

    nShifts = fld.(size(sy, 2), nBlocks)
    irotatedimsfcns = ([ t->ishiftdimspv(t, blk) for blk in nBlocks ]...,)
    ty = concatenateAtoms(nsolt, sy, nShifts, irotatedimsfcns; kwargs...)

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
            adjshiftcoefs!(Val(border), k, xu, xl, nshift)
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
            adjshiftcoefs!(Val(border), 2k, xum, xl, nshift)
            half_butterfly!(xu, xl)
            if nsolt.nChannels[1] < nsolt.nChannels[2]
                xu .= W' * xu
            else
                xl .= U' * xl
            end
            unnormalized_butterfly!(xu, xl)
            adjshiftcoefs!(Val(border), 2k-1, xu, xml, nshift)
            half_butterfly!(xu, xl)
        end
        px = rdfcn(px)
    end
    return px
end

ishiftFilterSymmetry(::Rnsolt, x::AbstractMatrix) = x

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
            xu .= W' * xu
            xl .= U' * xl

            B = getMatrixB(nsolt.nChannels, θ)
            px .= B' * px
            adjshiftcoefs!(Val(border), k, xu, xl, nshift)
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
        xW = @view px[1:fP, :]
        xU = @view px[(1:fP) .+ fP, :]
        xb = @view px[(fP+1):end, :]
        xŴ = @view px[1:cP, :]
        xÛ = @view px[(1:cP) .+ fP, :]
        xe  = @view px[1:end-1, :]

        params_d = (1:nstage, Wks, Uks, θ1ks, Ŵks, Ûks, θ2ks)
        foreach(reverse.(params_d)...) do k, W, U, θ1, Ŵ, Û, θ2
            xŴ .= Ŵ' * xŴ
            xÛ .= Û' * xÛ
            B2 = getMatrixB(nsolt.nChannels, θ2)
            xe .= B2' * xe
            adjshiftcoefs!(Val(border), 2k, xW, xb, nshift)
            xe .= B2 * xe

            xW .= W' * xW
            xU .= U' * xU
            B1 = getMatrixB(nsolt.nChannels, θ1)
            xe .= B1' * xe
            adjshiftcoefs!(Val(border), 2k-1, xW, xU, nshift)
            xe .= B1 * xe
        end
        px = rdfcn(px)
    end
    return px
end

ishiftFilterSymmetry(nsolt::Cnsolt, x::AbstractMatrix) = nsolt.Φ' * x

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
