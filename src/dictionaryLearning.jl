using ForwardDiff
using Base.Filesystem
using Statistics
using Dates

LearningTarget{N} = Union{CodeBook, NTuple{N, CodeBook}}

function train!(target::LearningTarget, trainingSet::AbstractArray; epochs::Integer=1, verbose::Union{Integer,Symbol}=1, logdir=Union{Nothing,AbstractString}=nothing, sc_options=(), du_options=())
    vlevel = verboselevel(verbose)

    savesettings(logdir, target, trainingSet;
        vlevel=vlevel,
        epochs=epochs,
        sc_options=sc_options,
        du_options=du_options)

    vlevel >= 1 && println("beginning dictionary training...")

    params_dic = getParamsDictionary(target)
    for itr = 1:epochs
        K = length(trainingSet)
        loss_sps = fill(Inf, K)
        loss_dus = fill(Inf, K)
        for k = 1:K
            x = trainingSet[k]

            vlevel >= 3 && println("start Sparse Coding Stage.")
            sparse_coefs, loss_sps[k] = stepSparseCoding(target, x; vlevel=vlevel, sc_options...)
            vlevel >= 3 && println("end Sparse Coding Stage.")

            vlevel >= 3 && println("start Dictionary Update.")
            params_dic, loss_dus[k] = updateDictionary(target, x, sparse_coefs, params_dic; vlevel=vlevel, du_options...)
            vlevel >= 3 && println("end Dictionary Update Stage.")

            vlevel >= 2 && println("epoch #$itr, data #$k: loss(Sparse coding) = $(loss_sps[k]), loss(Dic. update) = $(loss_dus[k]).")

            setParamsDictionary!(target, params_dic)
        end
        if vlevel >= 1
            println("--- epoch #$itr, sum(loss) = $(sum(loss_dus)), var(loss) = $(var(loss_dus))")
        end
        savelogs(logdir, target, itr;
            vlevel=vlevel,
            time=string(now()),
            loss_sparse_coding=loss_sps,
            loss_dictionary_update=loss_dus)
    end
    vlevel >= 1 && println("training finished.")
    return setParamsDictionary!(target, params_dic)
end

getParamsDictionary(nsolt::AbstractNsolt) = getrotations(nsolt)
setParamsDictionary!(nsolt::AbstractNsolt, pm::NTuple{2}) = setrotations!(nsolt, pm...)

function stepSparseCoding(cb::AbstractNsolt, x::AbstractArray; vlevel::Integer=0, sparsity=1.0, iterations::Integer=400, filter_domain::Symbol=:convolution, resource::AbstractResource=CPU1(FIR()), kwargs...)
    ana, syn = if filter_domain == :convolution
        ana = createAnalyzer(ConvolutionalOperator, cb, size(x); shape=Shapes.Vec())
        syn = createSynthesizer(ConvolutionalOperator, cb, size(x); shape=Shapes.Vec())
        (ana, syn)
    else # == :polyphase
        ana = createAnalyzer(cb, size(x); shape=Shapes.Vec())
        syn = createSynthesizer(cb, size(x); shape=Shapes.Vec())
        (ana, syn)
    end
    y0 = analyze(ana, x)

    # number of non-zero coefficients
    K = trunc(Int, sparsity * length(x))

    y_opt, loss_iht = iht(syn, ana, x, y0, K; iterations=iterations, isverbose=(vlevel>=3))

    return (y_opt, loss_iht)
end

updateDictionary(cb::CodeBook, x::AbstractArray, hy::AbstractArray; kwargs...) = updateDictionary(cb, x, hy, getParamsDictionary(cb), kwargs...)

function updateDictionary(nsolt::NS, x::AbstractArray, hy::AbstractArray, (θ, μ)::Tuple{TT,TM}; vlevel::Integer=0, stepsize::Real=1e-5, iterations::Integer=1, kwargs...) where {NS<:AbstractNsolt,TT<:AbstractArray,TM<:AbstractArray}
    f(t) = lossfcn(nsolt, x, hy, (t, μ))
    ∇f(t) = ForwardDiff.gradient(f, t)

    θopt = θ
    for itr = 1:iterations
        Δθ = ∇f(θopt)
        θopt -= stepsize * Δθ

        vlevel >= 3 && println("Dic. Up. Stage: #Iter. = $itr, ||∇loss|| (w.r.t. θ) = $(norm(Δθ))")
    end
    loss_opt = f(θopt)
    return ((θopt, μ), loss_opt,)
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

function lossfcn(nsolt::AbstractNsolt, x::AbstractArray, y::AbstractArray, (θ, μ)::Tuple{TT,TM}) where {TT<:AbstractArray,TM<:AbstractArray}
    cpnsolt = setrotations!(similar(nsolt, eltype(θ)), θ, μ)
    syn = createOperator(cpnsolt, size(x); shape=Shapes.Vec())
    norm(x - synthesize(syn, y))^2/2
end

namestring(nsolt::Rnsolt) = namestring(nsolt, "Real NSOLT")
namestring(nsolt::Cnsolt) = namestring(nsolt, "Complex NSOLT")
namestring(nsolt::AbstractNsolt{T,D}, strnsolt::AbstractString) where {T,D} = "$D-dimensional $(nsolt.category) $strnsolt"

verbosenames() = Dict(:none => 0, :standard => 1, :specified => 2, :loquacious => 3)
verboselevel(sym::Symbol) = verbosenames()[sym]
verboselevel(lv::Integer) = lv


getParamsDictionary(targets::NTuple) = getParamsDictionary.(targets)
setParamsDictionary!(targets::NTuple, pm::NTuple) = setParamsDictionary!.(targets, pm)

function stepSparseCoding(targets::NTuple{N,CB}, x::AbstractArray; vlevel::Integer=0, sparsity=1.0, iterations::Integer=400, kwargs...) where {N,CB<:CodeBook}
    ana = createAnalyzer(targets, size(x))
    syn = createSynthesizer(targets, size(x))
    y0 = analyze(ana, x)

    # number of non-zero coefficients
    K = trunc(Int, sparsity * length(x))

    y_opt, loss_iht = iht(syn, ana, x, y0, K; iterations=iterations, isverbose=(vlevel>=3))

    return (y_opt, loss_iht)
end

function updateDictionary(targets::NTuple{CB}, x::AbstractArray, hy::AbstractArray, (θ, μ)::Tuple{TT,TM}; vlevel::Integer=0, stepsize::Real=1e-5, iterations::Integer=1, kwargs...) where {CB<:CodeBook,TT<:AbstractArray,TM<:AbstractArray}
    f(t) = lossfcn(targets, x, hy, (t, μ))
    ∇f(t) = ForwardDiff.gradient(f, t)

    θopt = θ
    for itr = 1:iterations
        Δθ = ∇f(θopt)
        θopt -= stepsize * Δθ

        vlevel >= 3 && println("Dic. Up. Stage: #Iter. = $itr, ||∇loss|| (w.r.t. θ) = $(norm(Δθ))")
    end
    loss_opt = f(θopt)
    return ((θopt, μ), loss_opt,)
end

function savesettings(dirname::AbstractString, targets::NTuple{N,CB}, trainingset::AbstractArray; vlevel=0, epochs, sc_options=(), du_options=(), kwargs...) where {N,CB<:CodeBook,T,D}
    return nothing
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

function savelogs(dirname::AbstractString, targets::NTuple{N,CB}, epoch::Integer; params...) where {N,CB<:CodeBook}
    return nothing
    filename_logs = joinpath(dirname, "log")
    filename_nsolt = joinpath(dirname, "nsolt.json")

    strparams = [ " $(prm.first) = $(prm.second)," for prm in params ]
    strlogs = string("epoch $epoch:", strparams...)
    open(filename_logs, append=true) do io
        println(io, strlogs)
    end
    save(nsolt, filename_nsolt)
end

function lossfcn(targets::NTuple{N,CB}, x::AbstractArray, y::AbstractArray, (θ, μ)::Tuple{TT,TM}) where {N,CB<:CodeBook,TT<:AbstractArray,TM<:AbstractArray}
    cpnsolt = setrotations!(similar(nsolt, eltype(θ)), θ, μ)
    syn = createOperator(cpnsolt, size(x); shape=Shapes.Vec())
    norm(x - synthesize(syn, y))^2/2
end
