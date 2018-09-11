using ForwardDiff

function train!(nsolt::CB, trainingSet::AbstractArray; sparsity=1.0, epochs=1, verbose=:none, kwargs...) where {CB<:AbstractNsolt}
    θ, μ = getAngleParameters(nsolt)
    for itr = 1:epochs
        for k = 1:length(trainingSet)
            x = trainingSet[k]
            setAngleParameters!(nsolt, θ, μ)
            hy = stepSparseCoding(nsolt, x, sparsity)

            θ, μ = updateDictionary(nsolt, x, hy, θ, μ)

            if verbose != :none
                errk =
                println("epoch #$itr, data #$k: loss = $errk.")
            end
        end
        if verbose != :none
            println("---------- finish epoch #$itr ----------")
        end
    end
    if verbose != :none
        println("training finished.")
    end
    return setAngleParameters!(nsolt, θ, μ)
end

function stepSparseCoding(nsolt::AbstractNsolt, x::AbstractArray, sparsity; iterations::Integer=500)
    # afs = getAnalysisFilters(nsolt)
    # sfs = getSynthesisFilters(nsolt)
    # ana = createAnalyzer(afs; shape=:vector) # adjoint of Synthesizer
    # syn = createSynthesizer(syn; shape=:vector)
    ana = ConvolutionalOperator(:analyzer, nsolt, size(x); shape=:vector)
    syn = ConvolutionalOperator(:synthesizer, nsolt, size(x); shape=:vector)
    y0 = analyze(ana, x)

    K = trunc(Int, sparsity * length(x))

    iht(syn, ana, x, y0, K; iterations=iterations)
end

updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray) where {NS<:AbstractNsolt} = updateDictionary(nsolt, x, hy, getAngleParameters(nsolt)...)

function updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray, θ::AbstractArray, μ::AbstractArray) where {NS<:AbstractNsolt}
    df = nsolt.decimationFactor
    ord = nsolt.polyphaseOrder
    nch = nsolt.nChannels
    lossfcn(t) = begin
        syn = createSynthesizer(();shape=:vector)
        norm(x - synthesize(syn, hy))^2/2
    end
    ∇loss(t) = ForwardDiff.gradient(lossfcn, t)

    return (θ - η*∇loss(θ), μ)
end
