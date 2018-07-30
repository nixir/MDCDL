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
    L = length(angs)
    P = round(Integer, (1 + sqrt(1+8*L)) / 2)

    output = zeros(TA, L)
    nr = 1
    for idx1 = 1:P-1, idx2 = (idx1+1):P
        c, s = cos(angs[nr]), sin(angs[nr])
        sub1 = (idx1,fill(:,D-1)...)
        sub2 = (idx2,fill(:,D-1)...)

        ty1 = sig[idx1]*(-s * y[sub1...] + c * y[sub2...])
        ty2 = sig[idx2]*(-c * y[sub1...] - s * y[sub2...])
        output[nr] = vecdot(x[sub1...], ty1) + vecdot(x[sub2...], ty2)
        nr += 1
    end
    output
end

scalarGradOfOrthonormalMatrix(x::AbstractArray, y::AbstractArray, A::AbstractMatrix) = scalarGradOfOrthonormalMatrix(x, y, mat2rotations(A)...)
