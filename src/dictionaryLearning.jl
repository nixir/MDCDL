using ForwardDiff
using Base.Filesystem
using Statistics
using Dates

function train!(nsolt::CB, trainingSet::AbstractArray; epochs::Integer=1, verbose::Union{Integer,Symbol}=1, logdir=Union{Nothing,AbstractString}=nothing, sc_options=(), du_options=()) where {CB<:AbstractNsolt}
    vlevel = verboselevel(verbose)

    savesettings(logdir, nsolt, trainingSet;
        vlevel=vlevel,
        epochs=epochs,
        sc_options=sc_options,
        du_options=du_options)

    vlevel >= 1 && println("beginning NSOLT dictionary training...")

    θ, μ = getrotations(nsolt)
    for itr = 1:epochs
        K = length(trainingSet)
        loss_sps = fill(Inf, K)
        loss_dus = fill(Inf, K)
        for k = 1:K
            x = trainingSet[k]

            vlevel >= 3 && println("start Sparse Coding Stage.")
            hy, loss_sps[k] = stepSparseCoding(nsolt, x; vlevel=vlevel, sc_options...)
            vlevel >= 3 && println("end Sparse Coding Stage.")

            vlevel >= 3 && println("start Dictionary Update.")
            θ, μ, loss_dus[k] = updateDictionary(nsolt, x, hy, θ, μ; vlevel=vlevel, du_options...)
            vlevel >= 3 && println("end Dictionary Update Stage.")

            vlevel >= 2 && println("epoch #$itr, data #$k: loss(Sparse coding) = $(loss_sps[k]), loss(Dic. update) = $(loss_dus[k]).")

            setrotations!(nsolt, θ, μ)
        end
        if vlevel >= 1
            println("--- epoch #$itr, total sum(loss) = $(sum(loss_dus)), var(loss) = $(var(loss_dus))")
        end
        savelogs(logdir, nsolt, itr;
            vlevel=vlevel,
            time=string(now()),
            loss_sparse_coding=loss_sps,
            loss_dictionary_update=loss_dus)
    end
    vlevel >= 1 && println("training finished.")
    return setrotations!(nsolt, θ, μ)
end

function stepSparseCoding(nsolt::AbstractNsolt, x::AbstractArray; vlevel::Integer=0, sparsity=1.0, iterations::Integer=400, filter_domain::Symbol=:convolution, resource::AbstractResource=CPU1(FIR()), kwargs...)
    ana, syn = if filter_domain == :convolution
        (ConvolutionalOperator(nsolt, size(x), :analyzer; shape=:vector, resource=resource),
        ConvolutionalOperator(nsolt, size(x), :synthesizer; shape=:vector, resource=resource),)
    else # == :polyphase
        (createAnalyzer(nsolt, size(x); shape=:vector),
        createSynthesizer(nsolt, size(x); shape=:vector),)
    end
    y0 = analyze(ana, x)

    # number of non-zero coefficients
    K = trunc(Int, sparsity * length(x))

    y_opt, loss_iht = iht(syn, ana, x, y0, K; iterations=iterations, isverbose=(vlevel>=3))

    return (y_opt, loss_iht)
end

updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray; kwargs...) where {NS<:AbstractNsolt} = updateDictionary(nsolt, x, hy, getrotations(nsolt)..., kwargs...)

function updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray, θ::AbstractArray, μ::AbstractArray; vlevel::Integer=0, stepsize::Real=1e-5, iterations::Integer=1, kwargs...) where {NS<:AbstractNsolt}
    f(t) = lossfcn(nsolt, x, hy, t, μ)
    ∇f(t) = ForwardDiff.gradient(f, t)

    θopt = θ
    for itr = 1:iterations
        Δθ = ∇f(θopt)
        θopt -= stepsize * Δθ

        vlevel >= 3 && println("Dic. Up. Stage: #Iter. = $itr, ||∇loss|| (w.r.t. θ) = $(norm(Δθ))")
    end
    loss_opt = f(θopt)
    return (θopt, μ, loss_opt,)
end

savesettings(::Nothing, args...; kwargs...) = nothing
function savesettings(dirname::AbstractString, nsolt::AbstractNsolt{T,D}, trainingset::AbstractArray; vlevel=0, epochs, sc_options=(), du_options=(), kwargs...) where {T,D}
    filename = joinpath(dirname, "settings")

    txt_general = """
        Settings of $(namestring(nsolt)) Dictionary Learning:
        Element type := $T
        Decimation factor := $(nsolt.decimationFactor)
        Polyphase order := $(nsolt.polyphaseOrder)
        Number of channels := $(nsolt.nChannels)

        Number of training data := $(length(trainingset))
        Epochs := $epochs

        User-defined options for Sparse Coding Stage := $sc_options
        User-defined options for Dictionary Update Stage := $du_options

        """
    txt_others = [ "Keyword $(arg.first) := $(arg.second)\n" for arg in kwargs]

    open(filename, write=true) do io
        println(io, txt_general, txt_others...)
    end
    vlevel >= 2 && println("Settings was written in $filename.")
end

savelogs(::Nothing, args...; kwargs...) = nothing
function savelogs(dirname::AbstractString, nsolt::AbstractNsolt, epoch::Integer; params...)
    filename_logs = joinpath(dirname, "log")
    filename_nsolt = joinpath(dirname, "nsolt.json")

    strparams = [ " $(prm.first) = $(prm.second)," for prm in params ]
    strlogs = string("epoch $epoch:", strparams...)
    open(filename_logs, append=true) do io
        println(io, strlogs)
    end
    save(nsolt, filename_nsolt)
end

function lossfcn(nsolt::AbstractNsolt, x::AbstractArray, y::AbstractArray, θ::AbstractArray, μ::AbstractArray)
    cpnsolt = setrotations!(similar(nsolt, eltype(θ)), θ, μ)
    syn = createSynthesizer(cpnsolt, size(x); shape=:vector)
    norm(x - synthesize(syn, y))^2/2
end

namestring(nsolt::Rnsolt) = namestring(nsolt, "Real NSOLT")
namestring(nsolt::Cnsolt) = namestring(nsolt, "Complex NSOLT")
namestring(nsolt::AbstractNsolt{T,D}, strnsolt::AbstractString) where {T,D} = "$D-dimensional $(nsolt.category) $strnsolt"

verbosenames() = Dict(:none => 0, :standard => 1, :specified => 2, :loquacious => 3)
verboselevel(sym::Symbol) = verbosenames()[sym]
verboselevel(lv::Integer) = lv
