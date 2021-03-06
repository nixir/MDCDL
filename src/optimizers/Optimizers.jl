module Optimizers
    export iterations
    using NLopt

    abstract type AbstractOptimizer end
    abstract type AbstractGradientDescent <: AbstractOptimizer end
    struct Steepest <: AbstractGradientDescent
        ∇f
        iterations::Integer
        rate::AbstractFloat
        Steepest(∇f; iterations=1, rate=1e-3) = new(∇f, iterations, rate)
    end
    (stp::Steepest)(x0; kwargs...) = optimize(stp, x0; kwargs...)

    struct Momentum <: AbstractGradientDescent
        ∇f
        iterations::Integer
        rate::AbstractFloat
        β::AbstractFloat
        Momentum(∇f; iterations=1, rate=1e-3, beta=1e-8) = new(∇f, iterations, rate, beta)
    end
    (stp::Momentum)(x0; kwargs...) = optimize(stp, x0; kwargs...)

    struct AdaGrad <: AbstractGradientDescent
        ∇f
        iterations::Integer
        rate::AbstractFloat
        ϵ::AbstractFloat
        AdaGrad(∇f; iterations=1, rate=1e-3, epsilon=1e-8) = new(∇f, iterations, rate, epsilon)
    end
    (stp::AdaGrad)(x0; kwargs...) = optimize(stp, x0; kwargs...)

    struct Adam <: AbstractGradientDescent
        ∇f
        iterations::Integer
        rate::AbstractFloat
        β1::AbstractFloat
        β2::AbstractFloat
        ϵ::AbstractFloat

        Adam(∇f; iterations=1, rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8) = new(iterations, rate, beta1, beta2, epsilon)
    end
    (stp::Adam)(x0; kwargs...) = optimize(∇f, stp, x0; kwargs...)

    struct GlobalOpt <: AbstractOptimizer
        f
        alg::Symbol
        iterations::Integer
        xtolrel::Real
        bounds::NTuple{2}
        GlobalOpt(f; alg=:GN_MLSL_LDS, iterations=1, xtolrel=eps(), bounds=(-Float64(pi), Float64(pi))) = new(f, alg, iterations, xtolrel, bounds)
    end

    struct CRS <: AbstractOptimizer
        f
        iterations::Integer
        xtolrel::Real
        bounds::NTuple{2}
        CRS(f; iterations=1, xtolrel=eps(), bounds=(-Float64(pi), Float64(pi))) = new(f, iterations, xtolrel, bounds)
    end

    struct SGD <: AbstractGradientDescent
        ∇f
        iterations::Integer
        rate::Function
        SGD(∇f; iterations=1, rate=t->1e-3) = new(∇f, iterations, rate)
    end
    (stp::SGD)(x0; kwargs...) = optimize(stp, x0; kwargs...)

    iterations(abopt::AbstractOptimizer) = abopt.iterations

    function optimize(abopt::AbstractGradientDescent, x0::AbstractArray; isverbose::Bool=false, kwargs...)
        x = x0
        state = initialstate(abopt, x0; kwargs...)
        for k = 1:iterations(abopt)
            grad = gradient(abopt, x0)
            Δx, state = updatestep(abopt, grad, state, k; kwargs...)
            x = x - Δx

            # if norm(Δx) < eps; break
            # isverbose && println("#Iter. = $k, loss = $(f(vecpm_opt)), ||∇loss|| = $(norm(Δvecpm))")
        end
        x
    end

    gradient(abopt::AbstractOptimizer, x0) = abopt.∇f(x0)

    initialstate(::Steepest, args...; kwargs...) = ()
    initialstate(::AdaGrad, args...; kwargs...) = 0
    initialstate(adam::Adam, x0; kwargs...) = (zero(x0), 0)
    initialstate(sgd::SGD, args...; epoch=1, kwargs...) = (sgd.rate(epoch))

    updatestep(stp::Steepest, grad, args...; kwargs...) = (stp.rate .* grad, ())
    function updatestep(agd::AdaGrad, grad, v, itr; kwargs...)
        v = v + norm(grad)^2
        Δx = (agd.rate / (sqrt(v) + agd.ϵ)) * grad
        (upm, v)
    end
    function updatestep(adam::Adam, grad, (m, v), itr; kwargs...)
        m = adam.β1 * m + (1 - adam.β1) * grad
        v = adam.β2 * v + (1 - adam.β2) * norm(grad)^2
        m̂ = m / (1 - adam.β1^(itr))
        v̂ = v / (1 - adam.β2^(itr))

        Δx = adam.rate * m̂ / (sqrt(v̂) + adam.ϵ)
        (Δx, (m, v))
    end
    function updatestep(sgd::SGD, grad, rate, itr; kwargs...)
        (rate .* grad, rate)
    end

    function (gopt::GlobalOpt)(x0; kwargs...)
        opt = Opt(gopt.alg, length(x0))
        lower_bounds!(opt, gopt.bounds[1]*ones(size(x0)))
        upper_bounds!(opt, gopt.bounds[2]*ones(size(x0)))
        xtol_rel!(opt, gopt.xtolrel)
        xtol_abs!(opt, eps())
        maxeval!(opt, gopt.iterations)

        min_objective!(opt, gopt.f)
        minf, x_opt, ret = NLopt.optimize(opt, x0)

        return x_opt
    end

    function (gopt::CRS)(x0; kwargs...)
        opt = Opt(:GN_CRS2_LM, length(x0))
        lower_bounds!(opt, gopt.bounds[1]*ones(size(x0)))
        upper_bounds!(opt, gopt.bounds[2]*ones(size(x0)))
        xtol_rel!(opt, gopt.xtolrel)
        xtol_abs!(opt, eps())
        maxeval!(opt, gopt.iterations)

        min_objective!(opt, gopt.f)
        minf, x_opt, ret = NLopt.optimize(opt, x0)

        return x_opt
    end
end
