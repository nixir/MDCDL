using LinearAlgebra
using ColorTypes, ColorVectorSpace

# Fenchel-Moreau conjugate function
fmconj(f) = (x_, s_) -> (x_ - f(x_,s_))

# prox of l2-norm
proxL2(x::AbstractArray, λ::Real) = max(1.0 - λ/norm(x), 0) .* x

# prox. of mixed l1- and l2- norm
groupshrink(x, λ::Real) = proxL2.(x, λ)

# prox. of l1-norm
softshrink(x::AbstractArray, λ::Real) = softshrink.(x, λ)
softshrink(x, λ::Real) = max(1.0 - λ / abs(x), 0) * x

# prox. of nuclear norm.
function shrinkSingularValue(A::AbstractMatrix, λ::Real)
    X = svd(A, full=false)
    X.U * Diagonal(softshrink(X.S, λ)) * X.Vt
end

function boxProj(x::AbstractArray, lowerBound::Real, upperBound::Real)
    max.(min.(x,upperBound),lowerBound)
end

function l2ballProj(x::AbstractArray, radius::Real, c::AbstractArray)
    dist = norm(x - centerVec)

    ifelse(dist <= radius, x, (λ / dist) * (x - c) + c)
end

function hardshrink(x::AbstractArray, k::Integer; lt::Function=(lhs,rhs)->isless(abs2(lhs), abs2(rhs)))
    nzids = sortperm(vec(x); lt=lt, rev=true)[1:k]
    foldl(nzids; init=zero(x)) do mtx, idx
        setindex!(mtx, x[idx], idx)
    end
end

function hardshrink(x::AbstractArray, ks::AbstractArray)
    hardshrink.(x, ks)
end

function hardshrink(x::AbstractArray, ks::NTuple)
    hardshrink.(x, ks)
end
