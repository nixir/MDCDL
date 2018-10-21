using LinearAlgebra

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
