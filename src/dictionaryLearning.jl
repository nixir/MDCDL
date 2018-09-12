using ForwardDiff

function train!(nsolt::CB, trainingSet::AbstractArray; epochs::Integer=1, verbose::Union{Integer,Symbol}=1, sc_options=(), du_options=()) where {CB<:AbstractNsolt}
    dict_vlevels = Dict(:none => 0, :standard => 1, :specified => 2, :loquacious => 3)
    vlevel = if verbose isa Integer; verbose else dict_vlevels[verbose] end

    θ, μ = getAngleParameters(nsolt)
    for itr = 1:epochs
        K = length(trainingSet)
        lossvs = fill(Inf, K)
        for k = 1:K
            x = trainingSet[k]
            setAngleParameters!(nsolt, θ, μ)
            hy, loss_sp = stepSparseCoding(nsolt, x; sc_options...)

            θ, μ, loss_du = updateDictionary(nsolt, x, hy, θ, μ; du_options...)
            lossvs[k] = loss_du

            vlevel >= 2 && println("epoch #$itr, data #$k: loss(Sparse coding) = $loss_sp, loss(Dic. update) = $loss_du.")
        end
        if vlevel >= 1
            println("--- epoch #$itr, total loss = $(sum(lossvs))")
        end
    end
    vlevel >= 1 && println("training finished.")
    return setAngleParameters!(nsolt, θ, μ)
end

function stepSparseCoding(nsolt::AbstractNsolt, x::AbstractArray; sparsity=1.0, iterations::Integer=400, filter_domain::Symbol=:convolution, kwargs...)
    ana, syn = if filter_domain == :convolution
        (ConvolutionalOperator(:analyzer, nsolt, size(x); shape=:vector),
        ConvolutionalOperator(:synthesizer, nsolt, size(x); shape=:vector),)
    else # == :polyphase
        (createAnalyzer(nsolt, size(x); shape=:vector),
        createSynthesizer(nsolt, size(x); shape=:vector),)
    end
    y0 = analyze(ana, x)

    # number of non-zero coefficients
    K = trunc(Int, sparsity * length(x))

    y_opt, loss_iht = iht(syn, ana, x, y0, K; iterations=iterations)

    return (y_opt, loss_iht)
end

updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray) where {NS<:AbstractNsolt} = updateDictionary(nsolt, x, hy, getAngleParameters(nsolt)...)

function updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray, θ::AbstractArray, μ::AbstractArray; stepsize::Real=1e-5, iterations::Integer=1, kwargs...) where {NS<:AbstractNsolt}
    lossfcn(t) = begin
        cpnsolt = setAngleParameters!(similar(nsolt, eltype(t)), t, μ)
        syn = createSynthesizer(cpnsolt, size(x); shape=:vector)
        norm(x - synthesize(syn, hy))^2/2
    end
    ∇loss(t) = ForwardDiff.gradient(lossfcn, t)

    θopt = θ
    for itr = 1:iterations
        θopt -= stepsize * ∇loss(θ)
    end
    loss_opt = lossfcn(θopt)
    return (θopt, μ, loss_opt,)
end
