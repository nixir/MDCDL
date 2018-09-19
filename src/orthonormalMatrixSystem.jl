function mat2rotations(mtx::AbstractMatrix{T}) where T <: Real
    P = size(mtx, 1)

    res = similar(mtx, fld(P*(P-1),2))
    ids = [ (idx1, idx2) for idx1 = 1:P-1 for idx2 = (idx1+1):P ]

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

function rotations2mat(θs::AbstractArray, sig::AbstractArray)
    rotations2mat(θs, sig, round(Integer, (1 + sqrt(1+8*length(θs))) / 2))
end

function rotations2mat(θs::AbstractArray{TA}, sig::AbstractArray{TS}, P::Integer) where {TA<:Real,TS<:Number}
    mtx = Matrix{TA}(I,P,P)
    R = similar(mtx)

    ids = [ (idx1, idx2) for idx1 = 1:P-1 for idx2 = (idx1+1):P ]
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
