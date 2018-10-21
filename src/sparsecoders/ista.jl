using LinearAlgebra

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
