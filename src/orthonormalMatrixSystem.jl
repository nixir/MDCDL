function mat2rotations(mtx::AbstractMatrix{T}) where T <: Real
    sz = size(mtx)
    P = sz[1]

    res = Array{T}(undef, fld(P*(P-1),2))

    nr = 1
    for idx1 = 1:P-1, idx2 = (idx1+1):P
        a = givens(mtx, idx1, idx2, idx1)
        g = a[1]

        res[nr] = atan(g.s, g.c)
        nr += 1

        R = Matrix{T}(I,P,P)
        R[g.i1, g.i1] =  g.c
        R[g.i1, g.i2] =  g.s
        R[g.i2, g.i1] = -g.s
        R[g.i2, g.i2] =  g.c
        mtx = R*mtx
    end
    (res, round.(diag(mtx)))
end

function rotations2mat(θs::AbstractArray{TA}, sig::AbstractArray{TS}) where {TA<:Real,TS<:Number}
    L = length(θs)
    P = round(Integer, (1 + sqrt(1+8*L)) / 2)
    mtx = Matrix{TA}(I,P,P)

    nr = 1
    for idx1 = 1:P-1, idx2 = (idx1+1):P
        c = cos(θs[nr])
        s = sin(θs[nr])

        R = Matrix{TA}(I,P,P)
        R[idx1, idx1] =  c
        R[idx1, idx2] = -s
        R[idx2, idx1] =  s
        R[idx2, idx2] =  c
        mtx = mtx*R

        nr += 1
    end
    mtx * diagm(0 => sig)
end

# ∂(x'*A(θ)*y)/∂θ
function scalarGradOfOrthonormalMatrix(x::AbstractArray{TV,D}, y::AbstractArray{TV,D}, θs::AbstractArray{TA}, sig::AbstractArray{TS}) where {D,TV,TA<:Real, TS<:Number}
    scalarGradOfOrthonormalMatrix(x, y, rotations2mat(θs, sig))
end

function scalarGradOfOrthonormalMatrix(x::AbstractArray{TV,D}, y::AbstractArray{TV,D}, A::AbstractMatrix{TA}) where {D,TV,TA<:Real}
    θs, mus = mat2rotations(A)
    L = length(θs)
    P = round(Integer, (1 + sqrt(1+8*L)) / 2)

    ∇xᵀAy = Vector{TV}(undef, L)
    ay = A*y

    nr = 1
    for idx1 = 1:P-1, idx2 = (idx1+1):P
        c, s = cos(θs[nr]), sin(θs[nr])

        Rᵀ = Matrix{TA}(I,P,P)
        Rᵀ[idx1, idx1] =  c
        Rᵀ[idx1, idx2] =  s
        Rᵀ[idx2, idx1] = -s
        Rᵀ[idx2, idx2] =  c

        ∂R = zeros(TA,P,P)
        ∂R[idx1, idx1] = -s
        ∂R[idx1, idx2] = -c
        ∂R[idx2, idx1] =  c
        ∂R[idx2, idx2] = -s

        ay = Rᵀ * ay
        ∇xᵀAy[nr] = dot(x, ∂R * ay)
        x = Rᵀ * x

        nr += 1
    end
    ∇xᵀAy
end

# reference implementation
function scalarGradOfOrthonormalMatrix_reference(x::AbstractArray{TV,D}, y::AbstractArray{TV,D}, θs::AbstractArray{TA}, sig::AbstractArray{TS}) where {D,TV,TA<:Real, TS<:Number}
    L = length(θs)
    P = round(Integer, (1 + sqrt(1+8*L)) / 2)

    ids = filter((a) -> a[1] < a[2], vec([(i1, i2) for i2 = 1:P, i1 = 1:P]))

    rots = map(1:length(ids)) do nr
        i1, i2 = ids[nr][1], ids[nr][2]
        c, s = cos(θs[nr]), sin(θs[nr])

        R = Matrix{TA}(I,P,P)
        R[i1, i1] =  c
        R[i1, i2] = -s
        R[i2, i1] =  s
        R[i2, i2] =  c

        R
    end

    grots = map(1:length(ids)) do nr
        i1, i2 = ids[nr][1], ids[nr][2]
        c, s = cos(θs[nr]), sin(θs[nr])

        ∂R = zeros(TA,P,P)
        ∂R[i1, i1] = -s
        ∂R[i1, i2] = -c
        ∂R[i2, i1] =  c
        ∂R[i2, i2] = -s

        ∂R
    end

    erots = [ Matrix{TA}(I,P,P), rots..., Matrix{TA}(I,P,P) ]

    map(1:length(ids)) do nr
        real(dot(x, prod(erots[1:nr]) * grots[nr] * prod(erots[(nr+2):end]) * diagm(0 => sig) * y))
    end
end

scalarGradOfOrthonormalMatrix_reference(x::AbstractArray, y::AbstractArray, A::AbstractMatrix) = scalarGradOfOrthonormalMatrix_reference(x, y, mat2rotations(A)...)
