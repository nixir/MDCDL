using ForwardDiff
using Base.Filesystem
using Statistics
using Dates

LearningTarget{N} = Union{CodeBook, NTuple{N, CodeBook}}

function train!(target::LearningTarget, trainingSet::AbstractArray; epochs::Integer=1, sparsecoder::Symbol=:IHT, optimizer::Symbol=:SGD, verbose::Union{Integer,Symbol}=1, logdir=Union{Nothing,AbstractString}=nothing, sc_options=(), du_options=())
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
            sparse_coefs, loss_sps[k] = stepSparseCoding(Val(sparsecoder), target, x; vlevel=vlevel, sc_options...)
            vlevel >= 3 && println("end Sparse Coding Stage.")

            vlevel >= 3 && println("start Dictionary Update.")
            params_dic, loss_dus[k] = updateDictionary(Val(optimizer), target, x, sparse_coefs, params_dic; vlevel=vlevel, du_options...)
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

function stepSparseCoding(::Val{:IHT}, cb::DT, x::AbstractArray; vlevel::Integer=0, sparsity=1.0, iterations::Integer=400, filter_domain::Symbol=:convolution, resource::AbstractResource=CPU1(FIR()), kwargs...) where {DT<:LearningTarget}
    TP = getOperatorType_scs(DT, Val(filter_domain))

    ana = createAnalyzer(TP, cb, size(x); shape=Shapes.Vec())
    syn = createSynthesizer(TP, cb, size(x); shape=Shapes.Vec())

    # initial sparse vector y0
    y0 = analyze(ana, x)
    # number of non-zero coefficients
    K = trunc(Int, sparsity * length(x))

    y_opt, loss_iht = iht(syn, ana, x, y0, K; iterations=iterations, isverbose=(vlevel>=3))

    return (y_opt, loss_iht)
end

updateDictionary(::Val, cb::LearningTarget, x::AbstractArray, hy::AbstractArray; kwargs...) = updateDictionary(cb, x, hy, getParamsDictionary(cb), kwargs...)

function updateDictionary(::Val{:SGD}, cb::DT, x::AbstractArray, hy::AbstractArray, params; vlevel::Integer=0, stepsize::Real=1e-5, iterations::Integer=1, kwargs...) where {DT<:LearningTarget,TT<:AbstractArray,TM<:AbstractArray}
    vecpm, pminfo = decompose_params(DT, params)
    f(t) = lossfcn(cb, x, hy, compose_params(DT, t, pminfo))
    ∇f(t) = ForwardDiff.gradient(f, t)

    vecpm_opt = vecpm
    for itr = 1:iterations
        Δvecpm = ∇f(vecpm_opt)
        vecpm_opt -= stepsize * Δvecpm

        vlevel >= 3 && println("Dic. Up. Stage: #Iter. = $itr, ||∇loss|| = $(norm(Δvecpm))")
    end
    params_opt = compose_params(DT, vecpm_opt, pminfo)
    loss_opt = f(vecpm_opt)
    return (params_opt, loss_opt,)
end

savesettings(::Nothing, args...; kwargs...) = nothing
function savesettings(dirname::AbstractString, nsolt::AbstractNsolt{T,D}, trainingset::AbstractArray; vlevel=0, epochs, sc_options=(), du_options=(), filename="settings", kwargs...) where {T,D}
    filepath = joinpath(dirname, filename)

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

    open(filepath, write=true) do io
        println(io, txt_general, txt_others...)
    end
    vlevel >= 2 && println("Settings was written in $filename.")
    nothing
end

function savesettings(dirname::AbstractString, targets::NTuple{N,CB}, trainingset::AbstractArray; vlevel=0, epochs, sc_options=(), du_options=(), filename="settings", kwargs...) where {N,CB<:CodeBook,T,D}
    for idx = 1:N
        savesettings(dirname, targets[idx], trainingset; vlevel=vlevel, epochs=epochs, sc_options=sc_options, du_options=du_options, filename=string(filename,"_",idx))
    end
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
    nothing
end

function savelogs(dirname::AbstractString, targets::NTuple{N,CB}, epoch::Integer; params...) where {N,CB<:AbstractNsolt}
    for idx = 1:N
        filename_nsolt = joinpath(dirname, string("nsolt_", idx, ".json"))
        save(targets[idx], filename_nsolt)
    end
    return nothing
    # filename_logs = joinpath(dirname, "log")
    # filename_nsolt = joinpath(dirname, "nsolt.json")
    #
    # strparams = [ " $(prm.first) = $(prm.second)," for prm in params ]
    # strlogs = string("epoch $epoch:", strparams...)
    # open(filename_logs, append=true) do io
    #     println(io, strlogs)
    # end
    # save(nsolt, filename_nsolt)
end

function lossfcn(cb::LearningTarget, x::AbstractArray, y::AbstractArray, params) where {TT<:AbstractArray,TM<:AbstractArray}
    cpcb = similar_dictionary(cb, params)
    syn = createSynthesizer(cpcb, size(x); shape=Shapes.Vec())
    norm(x - synthesize(syn, y))^2/2
end

function similar_dictionary(nsolt::AbstractNsolt, (θ, μ)::Tuple{TT,TM}) where {TT<:AbstractArray,TM<:AbstractArray}
    setrotations!(similar(nsolt, eltype(θ)), θ, μ)
end

similar_dictionary(target::NTuple{N}, params::NTuple{N}) where {N} = similar_dictionary.(target, params)

getParamsDictionary(nsolt::AbstractNsolt) = getrotations(nsolt)
setParamsDictionary!(nsolt::AbstractNsolt, pm::NTuple{2}) = setrotations!(nsolt, pm...)

getParamsDictionary(targets::NTuple) = getParamsDictionary.(targets)
setParamsDictionary!(targets::NTuple, pm::NTuple) = setParamsDictionary!.(targets, pm)

getOperatorType_scs(::Type, ::Val) = AbstractOperator
getOperatorType_scs(::Type{NS}, ::Val{:convolution}) where {NS<:AbstractNsolt} = ConvolutionalOperator

decompose_params(::Type, params) = (params, ())
compose_params(::Type, params, ()) = params

decompose_params(::Type{NS}, (θ, μ)) where {NS<:AbstractNsolt} = (θ, μ)
compose_params(::Type{NS}, θ, μ) where {NS<:AbstractNsolt} = (θ, μ)

function decompose_params(::Type{NTuple{N,T}}, params::NTuple{N}) where {N,T<:AbstractNsolt}
    θs, μs = map(t->t[1], params), map(t->t[2], params)
    vcat(θs...), (length.(θs), μs)
end

function compose_params(::Type{NTuple{N,T}}, vp::AbstractArray, pminfo::Tuple) where {N,T<:AbstractNsolt}
    lenpms = collect(pminfo[1])
    arrpms = map(lenpms, cumsum(lenpms)) do lhs, rhs
        vp[(rhs - lhs + 1):rhs]
    end

    map((lhs,rhs)->(lhs,rhs,), (arrpms...,), pminfo[2])
end

namestring(nsolt::Rnsolt) = namestring(nsolt, "Real NSOLT")
namestring(nsolt::Cnsolt) = namestring(nsolt, "Complex NSOLT")
namestring(nsolt::AbstractNsolt{T,D}, strnsolt::AbstractString) where {T,D} = "$D-dimensional $(ifelse(istype1(nsolt), "Type-I", "Type-II")) $strnsolt"

verbosenames() = Dict(:none => 0, :standard => 1, :specified => 2, :loquacious => 3)
verboselevel(sym::Symbol) = verbosenames()[sym]
verboselevel(lv::Integer) = lv
