
struct IHT{T} <: AbstractSparseCoder
    x
    Φ
    Φᵀ
    iterations
    nonzeros::T

    absTol
    filter_domain::Symbol

    IHT(x, Φ, Φᵀ; iterations=nothing, nonzeros::T, absTol=1e2*eps(), filter_domain=:convolution) where {T<:Union{Integer,Tuple}} = new{T}(x, Φ, Φᵀ, iterations, nonzeros, absTol, filter_domain)
end

# argmin_{y} || x - Φy ||_2^2/2 s.t. ||y||_0 ≤ K
# Φ     : synthesis operator
# Φᵀ    : adjoint of Φ
#
function (iht::IHT)(y0; iterations=iht.iterations, isverbose::Bool=false)
    len = length(y0)
    yₖ = y0
    εx = Inf
    εy = Inf

    x̃ = iht.Φ(yₖ)
    for k = 1:iht.iterations
        yₖ₋₁ = yₖ

        yₖ = hardshrink(yₖ₋₁ + iht.Φᵀ(iht.x - x̃), iht.nonzeros)
        x̃ = iht.Φ(yₖ)

        εx = norm(iht.x - x̃)^2 / 2
        εy = norm(yₖ - yₖ₋₁)

        if isverbose
            println("number of Iterations $k: err = $εx, ||Δy|| = $εy.")
        end

        if εy <= iht.absTol
            break
        end
    end
    (yₖ, εx)
end
