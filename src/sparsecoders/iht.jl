
struct IHT{T} <: AbstractSparseCoder
    iterations::Integer
    nonzeros::T

    filter_domain::Symbol

    IHT(; iterations=1, nonzeros::T, filter_domain=:convolution) where {T<:Union{Integer,Tuple}} = new{T}(iterations, nonzeros, filter_domain)
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
