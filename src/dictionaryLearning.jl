using ForwardDiff

function train!(nsolt::CB, trainingSet::AbstractArray; epochs=1, verbose=1, sc_options=(), du_options=()) where {CB<:AbstractNsolt}
    dict_vlevels = Dict(:none => 0, :standard => 1, :specified => 2)
    vlevel = if verbose isa Integer; verbose else dict_vlevels[verbose] end

    θ, μ = getAngleParameters(nsolt)
    for itr = 1:epochs
        K = length(trainingSet)
        lossvs = fill(Inf, K)
        vlevel >= 1 && println("---------- begin epoch #$itr ----------")
        for k = 1:K
            x = trainingSet[k]
            setAngleParameters!(nsolt, θ, μ)
            hy, loss_sp = stepSparseCoding(nsolt, x; sc_options...)

            θ, μ, loss_du = updateDictionary(nsolt, x, hy, θ, μ; du_options...)
            lossvs[k] = loss_du

            vlevel >= 2 && println("epoch #$itr, data #$k: loss = $loss_du.")
        end
        if vlevel >= 1
            println("total loss = $(sum(lossvs))")
            println("---------- finish epoch #$itr ----------")
        end
    end
    vlevel >= 1 && println("training finished.")
    return setAngleParameters!(nsolt, θ, μ)
end

function stepSparseCoding(nsolt::AbstractNsolt, x::AbstractArray; sparsity=1.0, iterations::Integer=400, kwargs...)
    # ana = ConvolutionalOperator(:analyzer, nsolt, size(x); shape=:vector)
    # syn = ConvolutionalOperator(:synthesizer, nsolt, size(x); shape=:vector)
    ana = createAnalyzer(nsolt, size(x); shape=:vector)
    syn = createSynthesizer(nsolt, size(x); shape=:vector)
    y0 = analyze(ana, x)

    K = trunc(Int, sparsity * length(x))

    y_opt, loss_iht = iht(syn, ana, x, y0, K; iterations=iterations)

    return(y_opt, loss_iht)
end

updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray) where {NS<:AbstractNsolt} = updateDictionary(nsolt, x, hy, getAngleParameters(nsolt)...)

function updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray, θ::AbstractArray, μ::AbstractArray; step_size::Real=1e-5, kwargs...) where {NS<:AbstractNsolt}
    cpnsolt = deepcopy(nsolt)
    lossfcn(t) = begin
        syn = createSynthesizer(setAngleParameters!(cpnsolt, θ, μ), size(x); shape=:vector)
        norm(x - synthesize(syn, hy))^2/2
    end
    ∇loss(t) = ForwardDiff.gradient(lossfcn, t)

    hoge = ∇loss(θ)
    fuga = lossfcn(θ)
    @show hoge fuga
    θopt = θ - step_size*hoge
    loss_opt = lossfcn(θopt)
    return (θopt, μ, loss_opt,)
end
