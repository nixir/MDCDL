function mat2rotations(mtx0::AbstractMatrix{T}) where T <: Real
    sz = size(mtx0)
    P = sz[1]

    # res = Array{T}(undef, fld(P*(P-1),2))
    res = similar(mtx0, fld(P*(P-1),2))
    ids = [ (idx1, idx2) for idx1 = 1:P-1 for idx2 = (idx1+1):P ]

    mtx = Array(mtx0)
    for nr in 1:length(ids)
        a = givens(mtx, ids[nr][1], ids[nr][2], ids[nr][1])
        g = a[1]

        res[nr] = atan(g.s, g.c)

        R = Matrix{T}(I,P,P)
        R[g.i1, g.i1] =  g.c
        R[g.i1, g.i2] =  g.s
        R[g.i2, g.i1] = -g.s
        R[g.i2, g.i2] =  g.c
        mtx .= R*mtx
    end
    (res, round.(diag(mtx)))
end

function rotations2mat(θs::AbstractArray{TA}, sig::AbstractArray{TS}) where {TA<:Real,TS<:Number}
    L = length(θs)
    P = round(Integer, (1 + sqrt(1+8*L)) / 2)
    mtx = Matrix{TA}(I,P,P)

    ids = [ (idx1, idx2) for idx1 = 1:P-1 for idx2 = (idx1+1):P ]
    for nr in 1:length(ids)
        c, s = cos(θs[nr]), sin(θs[nr])
        idx1, idx2 = ids[nr][1], ids[nr][2]

        R = Matrix{TA}(I,P,P)
        R[idx1, idx1] =  c
        R[idx1, idx2] = -s
        R[idx2, idx1] =  s
        R[idx2, idx2] =  c

        mtx .= mtx*R
    end
    mtx * diagm(0 => sig)
end
