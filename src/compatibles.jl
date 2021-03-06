saivdr_permdctmtx(sz::Integer...) = saivdr_permdctmtx(sz)

saivdr_permdctmtx(sz::NTuple{1}) = permdctmtx(sz)

function saivdr_permdctmtx(sz::NTuple{2})
    T = Float64
    mtx = MDCDL.representationmatrix(x->dct(T.(x)), sz)

    # cls[1,1] = 1
    # cls[2,1] = 3
    # cls[1,2] = 4
    # cls[2,2] = 2
    cls = [ 1 4; 3 2 ]
    indexorders = map(CartesianIndices(sz)) do ci
        cls[(mod.(ci.I .- 1, 2) .+ 1)...]
    end
    permids = sortperm(vec(indexorders); alg=Base.DEFAULT_STABLE)

    @views vcat([ transpose(mtx[pi,:]) for pi in permids ]...)
end

# function saivdr_permdctmtx(sz::NTuple{3})
#     T = Float64
#     mtx = MDCDL.representationmatrix(x->dct(T.(x)), sz)
#
#     # cls[1,1,1] = 1
#     # cls[2,1,1] = 8
#     # cls[1,2,1] = 6
#     # cls[2,2,1] = 3
#     # cls[1,1,2] = 5
#     # cls[2,1,2] = 4
#     # cls[1,2,2] = 2
#     # cls[2,2,2] = 7
#     cls = cat([1 6; 8 3], [5 2; 4 7], dims=3)
#     indexorders = map(CartesianIndices(sz)) do ci
#         cls[(mod.(ci.I .- 1, 2) .+ 1)...]
#     end
#     permids = sortperm(vec(indexorders); alg=Base.DEFAULT_STABLE)
#
#     @views vcat([ transpose(mtx[pi,:]) for pi in permids ]...)
# end

function analyze_compatible(nsolt::AbstractNsolt{T,2}, px::AbstractMatrix, nBlocks::NTuple; kwargs...) where {T}
    ty = initialStep(nsolt, px; kwargs...)

    nShifts = fld.(size(px, 2), nBlocks)
    rotatedimsfcns = (t->rotatedimspv(t, nBlocks[1]), copy)
    exty = extendAtoms(nsolt, ty, nShifts, rotatedimsfcns; kwargs...)
    exty = rotatedimspv(exty, nBlocks[2])
    return shiftFilterSymmetry(nsolt, exty)
end

# function analyze_compatible(nsolt::AbstractNsolt{T,3}, px::AbstractMatrix, nBlocks::NTuple; kwargs...) where {T}
#     ty = initialStep(nsolt, px; kwargs...)
#
#     nShifts = fld.(size(px, 2), nBlocks)
#     rotatedimsfcns = (t->rotatedimspv(t, nShifts[1]), t->rotatedimspv(t, nShifts[2], copy)
#     exty = extendAtoms(nsolt, ty, nShifts, rotatedimsfcns; kwargs...)
#     exty = rotatedimspv(exty, nBlocks[3])
#     return shiftFilterSymmetry(nsolt, exty)
# end
