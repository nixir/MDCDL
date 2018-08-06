using ColorTypes
# import Base.LinAlg.norm
# import Base.LinAlg.vecnorm
# import Base.LinAlg.vecnormInf
# iterative shrinkage/thresholding algorithm
# solve a regularized convex optimization problem e.x. f(x) + g(x)
function ista(gradOfLossFcn::Function, prox::Function, stepSize::Real, x0; maxIterations::Integer=20, absTol::Real=1e2*eps(), viewFunction::Function=(itrs,tx,err)->nothing)
    x = x0
    errx = Inf
    len = length(x0)
    for nItr = 1:maxIterations
        xprev = x
        x = prox(x - stepSize*gradOfLossFcn(x), stepSize)

        errx = vecnorm(x-xprev)^2

        viewFunction(nItr, x, errx)

        if errx <= absTol
            break
        end
    end
    x
end

function fista(gradOfLossFcn::Function, prox::Function, stepSize::Real, x0; maxIterations::Integer=100, absTol::Real=1e2*eps(), viewFunction::Function=(itrs,tx,err)->nothing)
    x = x0
    errx = Inf
    len = length(x0)
    y = x0
    t = 1
    for nItr = 1:maxIterations
        xprev = x
        tprev = t

        x = prox(y - stepSize*gradOfLossFcn(y), stepSize)
        t = (1 + sqrt(1+4*t^2)) / 2
        y = x + (tprev-1)/t * (x - xprev)

        errx = vecnorm(x-xprev)^2

        viewFunction(nItr, x, errx)

        if errx <= absTol
            break
        end
    end
    x
end


# FB-based primal-dual splitting algorithm
# N. Komodakis and J. Pesquet, "Playing with Duality," IEEE Signal Processing Magazine, vol. 32, no. 6, pp. 31--54, 2015.
# Algorithm 3
# argmin_{x} f(x) + g(Lx) + h(x)
# f() and g(): proxiable function
# h(x): smooth convex function having a Lipschitzian gradient
# L: linear operator
function pds(gradOfLossFcn::Function, proxF::Function, proxG::Function, linearOperator::Function, adjOfLinearOperator::Function, τ::Real, σ::Real, x0, v0=linearOperator(x0); maxIterations::Integer=100, absTol::Real=1e-10, viewFunction::Function=(itrs,tx,err)->nothing)
    x = x0
    v = v0
    cproxG = (x, s) -> (x - proxG(x,s))
    for nItr = 1:maxIterations
        xprev = x
        p = proxF(x - τ*(gradOfLossFcn(x) + adjOfLinearOperator(v)), τ)
        q = cproxG(v + linearOperator(2.0*p - xprev), σ^-1)

        # (x, v) <- (x, v) + λ((p,q) - (x,v)) for　λ == 1
        x = p
        v = q

        errx = vecnorm(x - xprev)^2
        viewFunction(nItr, x, errx)
    end
end

function iht(synthesisFunc::Function, adjointSynthesisFunc::Function, x, y0, K; maxIterations::Integer=20, absTol::Real=1e-10, viewStatus::Bool=false, lt::Function=isless)
    len = length(y0)
    y = y0
    errx = Inf
    erry = Inf

    recx = synthesisFunc(y)
    for itr = 1:maxIterations
        yprev = y
        y = hardshrink(y + adjointSynthesisFunc(x - recx), K; lt=lt)
        recx = synthesisFunc(y)

        errx = vecnorm(x - recx)^2/2

        if viewStatus
            println("number of Iterations $itr: err = $errx ")
        end

        if errx <= absTol
            break
        end
    end
    y
end

function iht(cb::CodeBook, x, args...; kwargs...)
    iht((ty) -> synthesize(cb, ty, size(x)), (tx) -> adjoint_synthesize(cb, tx; outputMode=:vector), x, args...; kwargs...)
end

# prox of l2-norm
function proxOfL2(x, lambda::Real)
    max(1.0 - lambda/vecnorm(x), 0) * x
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
    dist = vecnorm(x - centerVec)

    if dist <= radius
        x
    else
        centerVec + lambda/dist*(x - centerVec)
    end
end

# hardshrink(x::AbstractArray{AbstractArray}, ks; kwargs...) = hardshrink.(x, ks; kwargs...)
function hardshrink(x::AbstractArray, k::Integer; lt::Function=isless)
    nzids = sortperm(vec(x); lt=lt, rev=true)[1:k]
    output = zeros(x)
    foreach(nzids) do idx
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
