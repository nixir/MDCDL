module Optimizers
    export iterations

    abstract type AbstractOptimizer end
    abstract type AbstractGradientDescent <: AbstractOptimizer end
    struct Steepest <: AbstractGradientDescent
        iterations::Integer
        rate::AbstractFloat
        Steepest(; iterations=1, rate=1e-3) = new(iterations, rate)
    end

    struct Momentum <: AbstractGradientDescent
        iterations::Integer
        rate::AbstractFloat
        β::AbstractFloat
        Momentum(; iterations=1, rate=1e-3, beta=1e-8) = new(iterations, rate, beta)
    end

    struct AdaGrad <: AbstractGradientDescent
        iterations::Integer
        rate::AbstractFloat
        ϵ::AbstractFloat
        AdaGrad(; iterations=1, rate=1e-3, epsilon=1e-8) = new(iterations, rate, epsilon)
    end

    struct Adam <: AbstractGradientDescent
        iterations::Integer
        rate::AbstractFloat
        β1::AbstractFloat
        β2::AbstractFloat
        ϵ::AbstractFloat

        Adam(; iterations=1, rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8) = new(iterations, rate, beta1, beta2, epsilon)
    end

    struct GlobalOpt <: AbstractOptimizer
        iterations::Integer
        xtolrel::Real
        GlobalOpt(; iterations=1, xtolrel=eps()) = new(iterations, xtolrel)
    end

    iterations(abopt::AbstractOptimizer) = abopt.iterations

    function optimize(abopt::AbstractGradientDescent, x0; kwargs...)

    end
end
