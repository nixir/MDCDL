givensids(P) = [ (idx1, idx2) for idx1 = 1:P-1 for idx2 = (idx1+1):P ]

ngivensangles(n::Integer) = fld(n*(n-1),2)

mat2rotations(args...) = mat2rotations_lowmemory(args...)

function mat2rotations_normal(mtx::AbstractMatrix{T}) where T <: Real
    P = size(mtx, 1)

    ids = givensids(P)
    res = similar(mtx, length(ids))
    mtx = Array(mtx)
    R = similar(mtx)
    for nr in 1:length(ids)
        a = givens(mtx, ids[nr][1], ids[nr][2], ids[nr][1])
        g = a[1]

        res[nr] = atan(g.s, g.c)

        R .= Matrix(I,P,P)
        R[g.i1, g.i1] =  g.c
        R[g.i1, g.i2] =  g.s
        R[g.i2, g.i1] = -g.s
        R[g.i2, g.i2] =  g.c
        mtx .= R*mtx
    end
    (res, round.(diag(mtx)))
end

function mat2rotations_lowmemory(mtx::AbstractMatrix{T}) where {T<:Real}
    ids = givensids(size(mtx, 1))
    res = similar(mtx, length(ids))

    mtx = Array(mtx)
    for (nr, (idx1, idx2)) in enumerate(ids)
        a = givens(mtx, idx1, idx2, idx1)
        g = a[1]

        res[nr] = atan(g.s, g.c)

        row1 = mtx[g.i1,:]
        mtx[g.i1,:] =  g.c * row1 + g.s * @view mtx[g.i2,:]
        mtx[g.i2,:] = -g.s * row1 + g.c * @view mtx[g.i2,:]
    end
    (res, round.(diag(mtx)))
end

function rotations2mat(θs::AbstractArray, sig::AbstractArray)
    rotations2mat(θs, sig, round(Integer, (1 + sqrt(1+8*length(θs))) / 2))
end

rotations2mat(θs, sig, P) = rotations2mat_lowmemory(θs, sig, P)

function rotations2mat_normal(θs::AbstractArray{TA}, sig::AbstractArray{TS}, P::Integer) where {TA<:Real,TS<:Number}
    mtx = Matrix{TA}(I,P,P)
    R = similar(mtx)

    ids = givensids(P)
    for nr in 1:length(ids)
        c, s = cos(θs[nr]), sin(θs[nr])
        idx1, idx2 = ids[nr]

        R .= Matrix(I,P,P)
        R[idx1, idx1] =  c
        R[idx1, idx2] = -s
        R[idx2, idx1] =  s
        R[idx2, idx2] =  c

        mtx .= mtx*R
    end
    mtx * diagm(0 => sig)
end

function rotations2mat_lowmemory(θs::AbstractArray{TA}, sig::AbstractArray{TS}, P::Integer) where {TA<:Real,TS<:Number}
    mtx = Matrix{TA}(I,P,P)

    for (nr, (idx1, idx2)) in enumerate(givensids(P))
        s, c = sincos(θs[nr])

        col1 = mtx[:,idx1]

        mtx[:, idx1] =  c * col1 + s * @view mtx[:,idx2]
        mtx[:, idx2] = -s * col1 + c * @view mtx[:,idx2]
    end
    mtx * diagm(0 => sig)
end

graph_one_factorization(P::Integer) = graph_one_factorization(Val(isodd(P)), P)

# When P is odd.
function graph_one_factorization(::Val{true}, P::Integer)
    map(0:P-1) do pivot_idx
        map(1:fld(P,2)) do offset
            ids = [ pivot_idx - offset, pivot_idx + offset ]
            (sort(mod.(ids, P) .+ 1)...,)
        end
    end
end

# When P is even.
function graph_one_factorization(::Val{false}, P::Integer)
    oddgf = graph_one_factorization(P-1)
    [ [ elem..., (idx, P) ] for (idx, elem) in enumerate(oddgf) ]
end

# function mat2rotations_permutated(mtx::AbstractMatrix{T}) where T <: Real
#     P = size(mtx, 1)
#
#     res = similar(mtx, fld(P*(P-1),2))
#     mtx = Array(mtx)
#
#     for (nr, (idx1, idx2)) in enumerate(vcat(graph_one_factorization(P)...))
#         a = givens(mtx, idx1, idx2, idx1)
#         g = a[1]
#
#         res[nr] = atan(g.s, g.c)
#
#         R = Matrix{T}(I,P,P)
#         R[g.i1, g.i1] =  g.c
#         R[g.i1, g.i2] =  g.s
#         R[g.i2, g.i1] = -g.s
#         R[g.i2, g.i2] =  g.c
#         mtx .= R*mtx
#     end
#     (res, round.(diag(mtx)))
# end
#
# function rotations2mat_permutated(θs::AbstractArray{TA}, sig::AbstractArray{TS}, P::Integer) where {TA<:Real,TS<:Number}
#     θrsp = reshape(θs, 2*cld(P,2)-1, fld(P,2))
#     mtx = mapreduce(*, enumerate(graph_one_factorization(P))) do (blk, ids)
#         R = Matrix{TA}(I,P,P)
#         for (nr, (idx1, idx2)) in enumerate(ids)
#             s = sin(θrsp[blk, nr])
#             c = cos(θrsp[blk, nr])
#             R[idx1, idx1] =  c
#             R[idx1, idx2] = -s
#             R[idx2, idx1] =  s
#             R[idx2, idx2] =  c
#         end
#         R
#     end
#     mtx * diagm(0 => sig)
# end
