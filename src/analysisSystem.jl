using ImageFiltering: imfilter, reflect, FIR, FFT
using OffsetArrays: OffsetArray

function analyze(A::TransformSystem{NS}, x::AbstractArray{TX,D}) where {TF,TX,D,NS<:AbstractNsolt{TF,D}}
    pvx = mdarray2polyphase(x, decimations(A.operator))
    y = analyze(A.operator, pvx; A.options...)
    reshape_polyvec(A.shape, A.operator, y)
end

analyze(fb::PolyphaseFB{TF,D}, pvx::PolyphaseVector{TX,D}; kwargs...) where {TF,TX,D} = PolyphaseVector(analyze(fb, pvx.data, pvx.nBlocks; kwargs...), pvx.nBlocks)

function analyze(nsolt::AbstractNsolt, px::AbstractMatrix, nBlocks::NTuple; kwargs...)
    ty = initialStep(nsolt, px; kwargs...)

    nShifts = fld.(size(px, 2), nBlocks)
    rotatedimsfcns = ([ t->rotatedimspv(t, blk) for blk in nBlocks ]...,)
    exty = extendAtoms(nsolt, ty, nShifts, rotatedimsfcns; kwargs...)
    return shiftFilterSymmetry(nsolt, exty)
end

# function analyze_compatible(nsolt::AbstractNsolt{T,2}, px::AbstractMatrix, nBlocks::NTuple; kwargs...) where {T}
#     ty = initialStep(nsolt, px; kwargs...)
#
#     nShifts = fld.(size(px, 2), nBlocks)
#     rotatedimsfcns = (t->copy(t), [ t->rotatedimspv(t, blk) for blk in nBlocks[1:end-1] ]...,)
#     exty = extendAtoms(nsolt, ty, nShifts, rotatedimsfcns; kwargs...)
#     exty = rotatedimspv(exty, nBlocks[end])
#     return shiftFilterSymmetry(nsolt, exty)
# end

function initialStep(nsolt::RnsoltTypeI, px::AbstractMatrix; kwargs...)
    M = prod(nsolt.decimationFactor)
    fM, cM = fld(M, 2), cld(M, 2)
    CJup = @view nsolt.CJ[1:cM, :]
    CJlw = @view nsolt.CJ[(1:fM) .+ cM, :]

    return [
        nsolt.W0 * Matrix(I, nsolt.nChannels[1], cM) * CJup * px;
        nsolt.U0 * Matrix(I, nsolt.nChannels[2], fM) * CJlw * px
    ]
end

function initialStep(nsolt::RnsoltTypeII, px::AbstractMatrix; kwargs...)
    M = prod(nsolt.decimationFactor)
    fM, cM = fld(M, 2), cld(M, 2)
    CJup = @view nsolt.CJ[1:cM, :]
    CJlw = @view nsolt.CJ[(1:fM) .+ cM, :]

    return [
        nsolt.W0 * Matrix(I, nsolt.nChannels[1], cM) * CJup * px;
        nsolt.U0 * Matrix(I, nsolt.nChannels[2], fM) * CJlw * px
    ]
end

function extendAtoms(nsolt::RnsoltTypeI, px::AbstractMatrix, nShifts::NTuple, rotatedimsfcns::Tuple; border=:circular)
    for d in nsolt.perm
        px = rotatedimsfcns[d](px)
        xu = @view px[1:nsolt.nChannels[1], :]
        xl = @view px[(1:nsolt.nChannels[2]) .+ nsolt.nChannels[1], :]

        foreach(1:nsolt.nStages[d], nsolt.Udks[d]) do k, U
            unnormalized_butterfly!(xu, xl)
            shiftcoefs!(Val(border), k, xu, xl, nShifts[d])
            half_butterfly!(xu, xl)

            xl .= U * xl
        end
    end
    return px
end

# TODO:
function extendAtoms(nsolt::RnsoltTypeII, px::AbstractMatrix, nShifts::NTuple, rotatedimsfcns; border=:circular)
    mnP, mxP = minmax(nsolt.nChannels...)
    # params = (rotatedimsfcns, nShifts, nsolt.nStages, nsolt.Wdks, nsolt.Udks)
    # foreach(params...) do rdfcn, nShifts[d], nstage, Wks, Uks
    for d in nsolt.perm
        px = rotatedimsfcns[d](px)

        xu = @view px[1:mnP, :]
        xl = @view px[end-mnP+1:end, :]

        xum = @view px[1:mxP, :]
        xml = @view px[end-mxP+1:end, :]

        foreach(1:nsolt.nStages[d], nsolt.Wdks[d], nsolt.Udks[d]) do k, W, U
            unnormalized_butterfly!(xu, xl)
            shiftcoefs!(Val(border), 2k-1, xu, xml, nShifts[d])
            half_butterfly!(xu, xl)
            if nsolt.nChannels[1] < nsolt.nChannels[2]
                xu .= W * xu
            else
                xl .= U * xl
            end

            unnormalized_butterfly!(xu, xl)
            shiftcoefs!(Val(border), 2k, xum, xl, nShifts[d])
            half_butterfly!(xu, xl)
            if nsolt.nChannels[1] < nsolt.nChannels[2]
                xml .= U * xml
            else
                xum .= W * xum
            end
        end
    end
    return px
end

shiftFilterSymmetry(::Rnsolt, y::AbstractMatrix) = y

function initialStep(nsolt::CnsoltTypeI, px::AbstractMatrix; kwargs...)
    Ipm = Matrix(I, nsolt.nChannels, prod(nsolt.decimationFactor))
    return nsolt.V0 * Ipm * nsolt.FJ * px
end

function initialStep(nsolt::CnsoltTypeII, px::AbstractMatrix; kwargs...)
    Ipm = Matrix(I, nsolt.nChannels, prod(nsolt.decimationFactor))
    return nsolt.V0 * Ipm * nsolt.FJ * px
end

function extendAtoms(nsolt::CnsoltTypeI, px::AbstractMatrix, nShifts::NTuple, rotatedimsfcns::NTuple; border=:circular)
    hP = fld(nsolt.nChannels, 2)
    for d in nsolt.perm
        px = rotatedimsfcns[d](px)
        xu = @view px[1:hP, :]
        xl = @view px[(1:hP) .+ hP, :]

        foreach(1:nsolt.nStages[d], nsolt.Wdks[d], nsolt.Udks[d], nsolt.θdks[d]) do k, W, U, θ
            B = getMatrixB(nsolt.nChannels, θ)
            px .= B' * px
            shiftcoefs!(Val(border), k, xu, xl, nShifts[d])
            px .= B * px

            xl .= U * xl
            xu .= W * xu
        end
    end
    return px
end

function extendAtoms(nsolt::CnsoltTypeII, px::AbstractMatrix, nShifts::NTuple, rotatedimsfcns; border=:circular)
    fP, cP = fld(nsolt.nChannels, 2), cld(nsolt.nChannels, 2)
    for d in nsolt.perm
        px = rotatedimsfcns[d](px)

        xW = @view px[1:fP, :]
        xU = @view px[(1:fP) .+ fP, :]
        xb = @view px[(fP+1):end, :]
        xŴ = @view px[1:cP, :]
        xÛ = @view px[(1:cP) .+ fP, :]
        xe  = @view px[1:end-1, :]

        foreach(1:nsolt.nStages[d], nsolt.Wdks[d], nsolt.Udks[d], nsolt.θ1dks[d], nsolt.Ŵdks[d], nsolt.Ûdks[d], nsolt.θ2dks[d]) do k, W, U, θ1, Ŵ, Û, θ2
            B1 = getMatrixB(nsolt.nChannels, θ1)
            xe .= B1' * xe
            shiftcoefs!(Val(border), 2k-1, xW, xU, nShifts[d])
            xe .= B1 * xe
            xU .= U * xU
            xW .= W * xW

            B2 = getMatrixB(nsolt.nChannels, θ2)
            xe .= B2' * xe
            shiftcoefs!(Val(border), 2k, xW, xb, nShifts[d])
            xe .= B2 * xe
            xÛ .= Û * xÛ
            xŴ .= Ŵ * xŴ
        end
    end
    return px
end

shiftFilterSymmetry(nsolt::Cnsolt, y::AbstractMatrix) = nsolt.Φ * y

function analyze(jts::JoinedTransformSystems{MS}, x::AbstractArray{TX,D}) where {MS<:Multiscale,TX,D}
    subanalyze(jts.shape, x, jts.transforms...)
end

# analyze(msop::MultiscaleOperator{TF,D}, x::AbstractArray{TX,D}) where {TF,TX,D} = subanalyze(msop.shape, x, msop.operators...)

# shape == Shapes.Separated
function subanalyze(shape::Shapes.Separated, sx::AbstractArray, abop::AbstractOperator, args...)
    sy = analyze(abop, sx)
    [sy[2:end], subanalyze(shape, sy[1], args...)...]
end

subanalyze(::Shapes.Separated, sx::AbstractArray, abop::AbstractOperator) = [ analyze(abop, sx) ]

# shape == Shapes.Combined
function subanalyze(shape::Shapes.Combined, sx::AbstractArray{T,D}, abop::AbstractOperator, args...) where {T,D}
    sy = analyze(abop, sx)
    clns = fill(:,D)
    [ sy[clns...,2:end], subanalyze(shape, sy[clns...,1], args...)... ]
end

subanalyze(::Shapes.Combined, sx::AbstractArray, abop::AbstractOperator) = [ analyze(abop, sx) ]

# shape == Shapes.Vec
function subanalyze(shape::Shapes.Vec, sx::AbstractArray, abop::AbstractOperator, args...)
    sy = analyze(abop, sx)
    lndc = fld(length(sy), nchannels(abop))
    dcdata = reshape(sy[1:lndc], args[1].shape.insize...)
    vcat(sy[lndc+1:end], subanalyze(shape, dcdata, args...))
end

subanalyze(::Shapes.Vec, sx::AbstractArray, abop::AbstractOperator) = analyze(abop, sx)

function analyze(pfs::ParallelFilters{TF,D}, x::AbstractArray{TX,D}; resource=CPU1(FIR())) where {TF,TX,D}
    df = pfs.decimationFactor
    ord = pfs.polyphaseOrder

    nShift = df .* fld.(ord, 2) .+ 1
    region = UnitRange.(1 .- nShift, df .* (ord .+ 1) .- nShift)
    offset = df .- 1

    map(pfs.kernelspair[1]) do f
        ker = reflect(OffsetArray(f, region...))
        fltimg = imfilter(resource, x, ker, "circular")
        downsample(fltimg, df, offset)
    end
end

function analyze(ca::TransformSystem{PF}, x::AbstractArray{TX,D}) where {TF,TX,D,PF<:ParallelFilters{TF,D}}
    y = analyze(ca.operator, x; ca.options...)
    reshape_polyvec(ca.shape, ca.operator, y)
end
