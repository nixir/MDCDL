using LinearAlgebra
using ColorTypes, ColorVectorSpace

struct ISTA
    ∇f
    prox
    η::Real
    iterations::Integer

    absTol::Real
end


function ISTA(∇f; iterations=nothing, prox=(t,λ)->t, η::Real=1.0, absTol=1e2*eps())
    ISTA(∇f, prox, η, iterations, absTol)
end

# iterative shrinkage/thresholding algorithm
# solve a regularized convex optimization problem e.x. f(x) + g(x)
# ∇loss : gradient of loss function f(x)
# prox  : proximity function of g(x)
# x0    : initial value
# η     : step size of update

function (ista::ISTA)(x0; iterations=ista.iterations, verboseFunction::Function=(args...)->nothing)
    xₖ = x0
    errx = Inf
    len = length(x0)
    for k = 1:ista.iterations
        xₖ₋₁ = xₖ
        xₖ = ista.prox(xₖ - ista.η*ista.∇f(xₖ), ista.η)

        errx = norm(xₖ - xₖ₋₁)^2 / 2

        verboseFunction(k, xₖ, errx)

        if errx <= ista.absTol
            break
        end
    end
    xₖ
end

struct FISTA
    ∇f
    prox
    η::Real
    iterations::Integer

    absTol::Real
end

function FISTA(∇f; iterations=nothing, prox=(t,λ)->t, η::Real=1.0, absTol=1e2*eps())
    FISTA(∇f, prox, η, iterations, absTol)
end

function (fista::FISTA)(x0; iterations=fista.iterations, verboseFunction::Function=(itrs,tx,err)->nothing)
    xₖ = x0
    errx = Inf
    len = length(x0)
    y = x0
    tₖ = 1
    for k = 1:fista.iterations
        xₖ₋₁, tₖ₋₁ = xₖ, tₖ

        xₖ = fista.prox(y - fista.η*fista.∇f(y), fista.η)
        tₖ = (1 + sqrt(1+4*tₖ^2)) / 2
        t̂ = (tₖ₋₁ - 1) / tₖ
        y = xₖ + t̂*(xₖ - xₖ₋₁)

        errx = norm(xₖ - xₖ₋₁)^2 / 2

        verboseFunction(k, xₖ, errx)

        if errx <= fista.absTol
            break
        end
    end
    xₖ
end

struct PDS
    ∇h
    proxF
    proxG
    L
    Lᵀ

    iterations::Integer
    τ::Real
    σ::Real
    λ::Real

    absTol::Real
end

function PDS(∇h; proxF=(t,a)->t, proxG=(t,a)->t, L=identity, adjL=identity, τ=1.0, σ=1.0, λ=1.0, iterations=nothing, absTol=1e2*eps())
    PDS(∇h, proxF, proxG, L, adjL, iterations, τ, σ, λ, absTol)
end

# FB-based primal-dual splitting algorithm
# N. Komodakis and J. Pesquet, "Playing with Duality," IEEE Signal Processing Magazine, vol. 32, no. 6, pp. 31--54, 2015.
# Algorithm 3
# argmin_{x} f(x) + g(Lx) + h(x)
# f() and g(): proxiable function
# h(x): smooth convex function having a Lipschitzian gradient
# ∇loss : gradient of loss function f(x)
# L: linear operator
# Lᵀ: adjoint of L
function (pds::PDS)(x0, v0=pds.L(x0); iterations=pds.iterations, verboseFunction::Function=(args...)->nothing)
    xₖ = x0
    v = v0
    cproxG = fmconj(pds.proxG)
    for k = 1:iterations
        xₖ₋₁ = xₖ
        p = pds.proxF(xₖ - pds.τ*(pds.∇h(xₖ) + pds.Lᵀ(v)), pds.τ)
        q = cproxG(v + pds.L(2.0*p - xₖ₋₁), pds.σ^-1)

        # (x, v) <- (x, v) + λ((p,q) - (x,v)) for　λ == 1
        xₖ, v = (x, v) + λ .* (p - x, q - v)
        # xₖ, v = p, q

        errx = norm(xₖ - xₖ₋₁)^2 /2
        verboseFunction(nItr, xₖ, errx)
        # TODO: break condition
    end
    xₖ
end

# argmin_{y} || x - Φy ||_2^2/2 s.t. ||y||_0 ≤ K
# Φ     : synthesis operator
# Φᵀ    : adjoint of Φ
#
function iht(Φ::Function, Φᵀ::Function, x, y0, S; iterations::Integer=1, absTol::Real=1e-10, isverbose::Bool=false)
    len = length(y0)
    yₖ = y0
    εx = Inf
    εy = Inf

    x̃ = Φ(yₖ)
    for k = 1:iterations
        yₖ₋₁ = yₖ

        yₖ = hardshrink(yₖ₋₁ + Φᵀ(x - x̃), S)
        x̃ = Φ(yₖ)

        εx = norm(x - x̃)^2 / 2
        εy = norm(yₖ - yₖ₋₁)

        if isverbose
            println("number of Iterations $k: err = $εx, ||Δy|| = $εy.")
        end

        if εy <= absTol
            break
        end
    end
    (yₖ, εx)
end

# Fenchel-Moreau conjugate function
fmconj(f) = (x_, s_) -> (x_ - f(x_,s_))

function iht(cb::CodeBook, x, args...; kwargs...)
    syn = createSynthesizer(cb, x; shape=Shapes.Vec())
    adj = createAnalyzer(cb, x; shape=Shapes.Vec())
    iht(syn, adj, x, args...; kwargs...)
end

function iht(a::AbstractOperator, s::AbstractOperator, x, args...; kwargs...)
    iht(t->synthesize(a, t), t->analyze(s, t), x, args...; kwargs...)
end

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
