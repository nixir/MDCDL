using LinearAlgebra
using ColorTypes
# iterative shrinkage/thresholding algorithm
# solve a regularized convex optimization problem e.x. f(x) + g(x)
# ∇loss : gradient of loss function f(x)
# prox  : proximity function of g(x)
# x0    : initial value
# η     : step size of update
function ista(∇loss::Function, prox::Function, x0; η::Real=1.0, iterations::Integer=20, absTol::Real=1e2*eps(), verboseFunction::Function=(args...)->nothing)
    xₖ = x0
    errx = Inf
    len = length(x0)
    for k = 1:iterations
        xₖ₋₁ = xₖ
        xₖ = prox(xₖ - η*∇loss(xₖ), η)

        errx = norm(xₖ - xₖ₋₁)^2 / 2

        verboseFunction(k, xₖ, errx)

        if errx <= absTol
            break
        end
    end
    xₖ
end

function fista(∇loss::Function, prox::Function, x0; η::Real=1.0, iterations::Integer=100, absTol::Real=1e2*eps(), verboseFunction::Function=(itrs,tx,err)->nothing)
    xₖ = x0
    errx = Inf
    len = length(x0)
    y = x0
    tₖ = 1
    for k = 1:iterations
        xₖ₋₁, tₖ₋₁ = xₖ, tₖ

        xₖ = prox(y - η*∇loss(y), η)
        tₖ = (1 + sqrt(1+4*tₖ^2)) / 2
        t̂ = (tₖ₋₁ - 1) / tₖ
        y = xₖ + t̂*(xₖ - xₖ₋₁)

        errx = norm(xₖ - xₖ₋₁)^2 / 2

        verboseFunction(k, xₖ, errx)

        if errx <= absTol
            break
        end
    end
    xₖ
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
function pds(∇loss::Function, proxF::Function, proxG::Function, L::Function, Lᵀ::Function, x0, v0=L(x0); τ::Real=1.0, σ::Real=1.0, iterations::Integer=100, absTol::Real=1e-10, verboseFunction::Function=(args...)->nothing)
    xₖ = x0
    v = v0
    # cproxG = (x_, s_) -> (x_ - proxG(x_,s_))
    cproxG = fmconj(proxG)
    for k = 1:iterations
        xₖ₋₁ = xₖ
        p = proxF(xₖ - τ*(∇loss(xₖ) + Lᵀ(v)), τ)
        q = cproxG(v + L(2.0*p - xₖ₋₁), σ^-1)

        # (x, v) <- (x, v) + λ((p,q) - (x,v)) for　λ == 1
        xₖ, v = p, q

        errx = norm(xₖ - xₖ₋₁)^2 /2
        verboseFunction(nItr, xₖ, errx)
    end
    xₖ
end

# argmin_{y} || x - Φy ||_2^2/2 s.t. ||y||_0 ≤ K
# Φ     : synthesis operator
# Φᵀ    : adjoint of Φ
#
function iht(Φ::Function, Φᵀ::Function, x, y0, S; iterations::Integer=1, absTol::Real=1e-10, isverbose::Bool=false, lt::Function=isless)
    len = length(y0)
    yₖ = y0
    εx = Inf
    # εy = Inf

    x̃ = Φ(yₖ)
    for k = 1:iterations
        yₖ₋₁ = yₖ

        yₖ = hardshrink(yₖ₋₁ + Φᵀ(x - x̃), S; lt=lt)
        x̃ = Φ(yₖ)

        εx = norm(x - x̃)^2 / 2

        if isverbose
            println("number of Iterations $k: err = $errx ")
        end

        if εx <= absTol
            break
        end
    end
    yₖ
end

# function iht(cb::CodeBook, x, args...; kwargs...)
#     iht((ty) -> synthesize(cb, ty, size(x)), (tx) -> adjoint_synthesize(cb, tx; shape=:vector), x, args...; kwargs...)
# end

# Fenchel-Moreau conjugate function
fmconj(f) = (x_, s_) -> (x_ - f(x_,s_))

function iht(cb::CodeBook, x, args...; kwargs...)
    syn = createSynthesizer(cb, x; shape=:vector)
    adj = syn'
    iht(syn, adj, x, args...; kwargs...)
end

function iht(a::AbstractSynthesizer, s::AbstractAnalyzer, x, args...; kwargs...)
    iht(t->synthesize(a, t), t->analyze(s, t), x, args...; kwargs...)
end

# prox of l2-norm
function proxOfL2(x, lambda::Real)
    max(1.0 - lambda/norm(x), 0) .* x
end

# prox. of mixed l1- and l2- norm
function groupshrink(x, lambda::Real)
    proxOfL2.(x, lambda)
end

# prox. of l1-norm
softshrink(x::AbstractArray{T}, lambda::Real) where T = softshrink.(x, lambda)
function softshrink(x, lambda::Real)
    max(1.0 - lambda / abs(x),0) * x
end

# prox. of nuclear norm.
function shrinkSingularValue(A::Matrix, lambda::Real)
    U, S, V = svd(A,thin=true)
    U * diagm(softshrink(S,lambda)) * V'
end

function boxProj(x, lowerBound::Real, upperBound::Real)
    max.(min.(x,upperBound),lowerBound)
end

function l2ballProj(x::T, radius::Real, centerVec::T) where T
    dist = norm(x - centerVec)

    if dist <= radius
        x
    else
        centerVec + lambda/dist*(x - centerVec)
    end
end

# hardshrink(x::AbstractArray{AbstractArray}, ks; kwargs...) = hardshrink.(x, ks; kwargs...)
function hardshrink(x::AbstractArray, k::Integer; lt::Function=isless)
    nzids = sortperm(vec(x); lt=lt, rev=true)[1:k]
    output = zero(x)
    for idx in nzids
        output[idx] = x[idx]
    end
    output
end

# function hardshrink(x::AbstractArray{Complex{T}}, ks::Integer) where T
#     hardshrink(x, ks; lt=(lhs,rhs)->isless(abs(lhs),abs(rhs)))
# end
#
# function hardshrink(x::AbstractArray{Colorant}, ks::Integer)
#     normc = (c) -> mapreducec((v)->v^2,+,0,c)
#     ltfcn = (lhs,rhs)->isless(normc(lhs),normc(rhs))
#     println("FOOOOOOO")
#     hardshrink(x, ks; lt=ltfcn)
# end
