using ForwardDiff
using Base.Filesystem
using Statistics
using Dates

function train!(nsolt::CB, trainingSet::AbstractArray; epochs::Integer=1, verbose::Union{Integer,Symbol}=1, logdir=Union{Nothing,AbstractString}=nothing, sc_options=(), du_options=()) where {CB<:AbstractNsolt}
    dict_vlevels = Dict(:none => 0, :standard => 1, :specified => 2, :loquacious => 3)
    vlevel = if verbose isa Integer; verbose else dict_vlevels[verbose] end

    θ, μ = getrotations(nsolt)
    for itr = 1:epochs
        K = length(trainingSet)
        loss_sps = fill(Inf, K)
        loss_dus = fill(Inf, K)
        for k = 1:K
            x = trainingSet[k]

            hy, loss_sps[k] = stepSparseCoding(nsolt, x; sc_options...)
            θ, μ, loss_dus[k] = updateDictionary(nsolt, x, hy, θ, μ; du_options...)

            vlevel >= 2 && println("epoch #$itr, data #$k: loss(Sparse coding) = $(loss_sps[k]), loss(Dic. update) = $(loss_dus[k]).")

            setrotations!(nsolt, θ, μ)
        end
        if vlevel >= 1
            println("--- epoch #$itr, total sum(loss) = $(sum(loss_dus)), var(loss) = $(var(loss_dus))")
        end
        log_params = [  "time" => Dates.format(now(), "yyy_mm_dd_SS_sss"),
                        "loss_sparse_coding" => loss_sps,
                        "loss_dictionary_update" => loss_dus ]
        savelogs(logdir, nsolt, itr, log_params...)
    end
    vlevel >= 1 && println("training finished.")
    return setrotations!(nsolt, θ, μ)
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

updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray) where {NS<:AbstractNsolt} = updateDictionary(nsolt, x, hy, getrotations(nsolt)...)

function updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray, θ::AbstractArray, μ::AbstractArray; stepsize::Real=1e-5, iterations::Integer=1, kwargs...) where {NS<:AbstractNsolt}
    lossfcn(t) = begin
        cpnsolt = setrotations!(similar(nsolt, eltype(t)), t, μ)
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

savelogs(::Nothing, args...; kwargs...) = nothing
function savelogs(dirname::AbstractString, nsolt::AbstractNsolt, epoch::Integer, params::Pair{String}...)
    filename_logs = joinpath(dirname, "log")
    filename_nsolt = joinpath(dirname, "nsolt.json")

    strparams = map(prm->" $(prm.first) = $(prm.second),", params)
    strlogs = string("epoch $epoch:", strparams...)
    open(filename_logs, append=true) do io
        println(io, strlogs)
    end
    save(nsolt, filename_nsolt)
end
