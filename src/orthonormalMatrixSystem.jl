function mat2rotations(mtx::Matrix{T}) where T <: Real
    sz = size(mtx)
    P = sz[1]

    res = Array{T}(fld(P*(P-1),2))

    nr = 1
    for idx1 = 1:P-1, idx2 = (idx1+1):P
        a = givens(mtx, idx1, idx2, idx1)
        g = a[1]

        res[nr] = atan2(g.s,g.c)
        nr += 1

        rtm = eye(T,P)
        rtm[g.i1, g.i1] =  g.c
        rtm[g.i1, g.i2] =  g.s
        rtm[g.i2, g.i1] = -g.s
        rtm[g.i2, g.i2] =  g.c
        mtx = rtm*mtx
    end
    (res, round.(diag(mtx)))
end

function rotations2mat(angs::Array{TA}, sig::Array{TS}) where {TA<:Real,TS<:Number}
    L = length(angs)
    P = round(Integer, (1 + sqrt(1+8*L)) / 2)
    mtx = eye(TA,P)

    nr = 1
    for idx1 = 1:P-1, idx2 = (idx1+1):P
        c = cos(angs[nr])
        s = sin(angs[nr])

        rtm = eye(TA,P)
        rtm[idx1, idx1] =  c
        rtm[idx1, idx2] = -s
        rtm[idx2, idx1] =  s
        rtm[idx2, idx2] =  c
        mtx = mtx*rtm

        nr += 1
    end
    mtx * diagm(sig)
end

# ∂(x'*A(θ)*y)/∂θ
function scalarGradOfOrthonormalMatrix(x::AbstractArray{TV,D}, y::AbstractArray{TV,D}, angs::Array{TA}, sig::Array{TS}) where {D,TV,TA<:Real, TS<:Number}
    scalarGradOfOrthonormalMatrix_reference(x, y, angs, sig)
end

scalarGradOfOrthonormalMatrix(x::AbstractArray, y::AbstractArray, A::AbstractMatrix) = scalarGradOfOrthonormalMatrix(x, y, mat2rotations(A)...)

function scalarGradOfOrthonormalMatrix_reference(x::AbstractArray{TV,D}, y::AbstractArray{TV,D}, angs::Array{TA}, sig::Array{TS}) where {D,TV,TA<:Real, TS<:Number}
    L = length(angs)
    P = round(Integer, (1 + sqrt(1+8*L)) / 2)

    ids = filter((a) -> a[1] < a[2], vec([(i1, i2) for i2 = 1:P, i1 = 1:P]))

    rots = map(1:length(ids)) do nr
        i1, i2 = ids[nr][1], ids[nr][2]
        c, s = cos(angs[nr]), sin(angs[nr])

        rtm = eye(TA,P)
        rtm[i1, i1] =  c
        rtm[i1, i2] = -s
        rtm[i2, i1] =  s
        rtm[i2, i2] =  c

        rtm
    end

    grots = map(1:length(ids)) do nr
        i1, i2 = ids[nr][1], ids[nr][2]
        c, s = cos(angs[nr]), sin(angs[nr])

        grtm = zeros(TA,P,P)
        grtm[i1, i1] = -s
        grtm[i1, i2] = -c
        grtm[i2, i1] =  c
        grtm[i2, i2] = -s

        grtm
    end

    erots = [ eye(TA,P), rots..., eye(TA,P) ]

    map(1:length(ids)) do nr
        vecdot(x, prod(erots[1:nr]) * grots[nr] * prod(erots[(nr+2):end]) * diagm(sig) * y)
    end
end

scalarGradOfOrthonormalMatrix_reference(x::AbstractArray, y::AbstractArray, A::AbstractMatrix) = scalarGradOfOrthonormalMatrix_reference(x, y, mat2rotations(A)...)
