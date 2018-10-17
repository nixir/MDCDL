using ForwardDiff
using Base.Filesystem
using Statistics
using Dates
using FileIO: load
using ImageCore: channelview
using NLopt

module SparseCoders
    abstract type AbstractSparseCoder end
    struct IHT{T} <: AbstractSparseCoder
        iterations::Integer
        nonzeros::T

        filter_domain::Symbol

        IHT(; iterations=1, nonzeros::T, filter_domain=:convolution) where {T<:Union{Integer,Tuple}} = new{T}(iterations, nonzeros, filter_domain)
    end
end

module Optimizers
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
end

iterations(abopt::Optimizers.AbstractOptimizer) = abopt.iterations

LearningTarget{N} = Union{CodeBook, NTuple{N, CodeBook}, Multiscale}

function train!(target::LearningTarget, trainingSet::AbstractArray;
    epochs::Integer=1, shape=nothing,
    sparsecoder::SparseCoders.AbstractSparseCoder=SparseCoders.IHT(),
    optimizer::Optimizers.AbstractOptimizer=Optimizers.Steepest(),
    verbose::Union{Integer,Symbol}=1, logdir=Union{Nothing,AbstractString}=nothing,
    plot_function=t->nothing)

    vlevel = verboselevel(verbose)

    savesettings(logdir, target, trainingSet;
        vlevel=vlevel,
        epochs=epochs)

    vlevel >= 1 && println("beginning dictionary training...")

    params_dic = getParamsDictionary(target)
    for itr = 1:epochs
        K = length(trainingSet)
        loss_sps = fill(Inf, K)
        loss_dus = fill(Inf, K)
        for k = 1:K
            x = gettrainingdata(trainingSet[k])
            shapek = getvalidshape(shape, target, sparsecoder, optimizer, x)

            vlevel >= 3 && println("start Sparse Coding Stage.")
            sparse_coefs, loss_sps[k] = stepSparseCoding(sparsecoder, target, x; shape=shapek, vlevel=vlevel)
            vlevel >= 3 && println("end Sparse Coding Stage.")

            vlevel >= 3 && println("start Dictionary Update.")
            params_dic, loss_dus[k] = updateDictionary(optimizer, target, x, sparse_coefs, params_dic; shape=shapek, vlevel=vlevel)
            vlevel >= 3 && println("end Dictionary Update Stage.")

            vlevel >= 2 && println("epoch #$itr, data #$k: loss(Sparse coding) = $(loss_sps[k]), loss(Dic. update) = $(loss_dus[k]).")

            setParamsDictionary!(target, params_dic)
        end
        if vlevel >= 1
            println("--- epoch #$itr, sum(loss) = $(sum(loss_sps)), var(loss) = $(var(loss_sps))")
        end
        # plot_function(target)

        savelogs(logdir, target, itr;
            vlevel=vlevel,
            time=string(now()),
            loss_sparse_coding=loss_sps,
            loss_dictionary_update=loss_dus)
    end
    vlevel >= 1 && println("training finished.")
    return setParamsDictionary!(target, params_dic)
end

function stepSparseCoding(ihtsc::SparseCoders.IHT, cb::DT, x::AbstractArray; shape::Shapes.AbstractShape=Shapes.Vec(size(x)), vlevel::Integer=0, kwargs...) where {DT<:LearningTarget}
    ts = createTransform(cb, shape)

    # initial sparse vector y0
    y0 = analyze(ts, x)
    # number of non-zero coefficients

    y_opt, loss_iht = iht(ts, ts, x, y0, ihtsc.nonzeros; iterations=ihtsc.iterations, isverbose=(vlevel>=3))

    return (y_opt, loss_iht)
end

updateDictionary(::Optimizers.AbstractOptimizer, cb::LearningTarget, x::AbstractArray, hy::AbstractArray; kwargs...) = updateDictionary(cb, x, hy, getParamsDictionary(cb), kwargs...)

function updateDictionary(optr::Optimizers.AbstractGradientDescent, cb::DT, x::AbstractArray, hy::AbstractArray, params; shape::Shapes.AbstractShape=Shapes.Vec(size(x)), vlevel::Integer=0, kwargs...) where {DT<:LearningTarget,TT<:AbstractArray,TM<:AbstractArray}
    vecpm, pminfo = decompose_params(DT, params)
    f(t) = lossfcn(cb, x, hy, compose_params(DT, t, pminfo); shape=shape)
    ∇f(t) = ForwardDiff.gradient(f, t)

    vecpm_opt = vecpm
    optim_state = initialstate(optr, ∇f, vecpm)
    for itr = 1:iterations(optr)
        Δvecpm = ∇f(vecpm_opt)
        upm, optim_state = updateamount(optr, Δvecpm, itr, optim_state)
        vecpm_opt -= upm

        vlevel >= 3 && println("Dic. Up. Stage: #Iter. = $itr, loss = $(f(vecpm_opt)), ||∇loss|| = $(norm(Δvecpm))")
    end
    params_opt = compose_params(DT, vecpm_opt, pminfo)
    loss_opt = f(vecpm_opt)
    return (params_opt, loss_opt,)
end

function updateDictionary(nlopt::Optimizers.GlobalOpt, cb::DT, x::AbstractArray, hy::AbstractArray, params; shape=Shapes.Vec(size(x)), vlevel::Integer=0, kwargs...) where {DT<:LearningTarget}
    vecpm, pminfo = decompose_params(DT, params)
    f(t, grad=nothing) = lossfcn(cb, x, hy, compose_params(DT, t, pminfo); shape=shape)

    opt = Opt(:GN_MLSL_LDS, length(vecpm))
    lower_bounds!(opt, -1*pi*ones(size(vecpm)))
    upper_bounds!(opt,  1*pi*ones(size(vecpm)))
    xtol_rel!(opt, nlopt.xtolrel)
    xtol_abs!(opt, eps())
    maxeval!(opt, nlopt.iterations)

    min_objective!(opt, f)
    minf, vecpm_opt, ret = optimize(opt, vecpm)
    params_opt = compose_params(DT, vecpm_opt, pminfo)
    # vlevel >= 0 && println("fopt(θ) = $minf, f(θ) = $(f(minx))")
    return (params_opt, minf)
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

function savesettings(dirname::AbstractString, targets::Multiscale, trainingset::AbstractArray; vlevel=0, epochs, sc_options=(), du_options=(), filename="settings", kwargs...) where {T,D}
    for idx = 1:length(targets)
        savesettings(dirname, targets.filterbanks[idx], trainingset; vlevel=vlevel, epochs=epochs, sc_options=sc_options, du_options=du_options, filename=string(filename,"_",idx))
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
    savefb(filename_nsolt, nsolt)
    nothing
end

function savelogs(dirname::AbstractString, targets::Multiscale, epoch::Integer; params...) # where {N,CB<:AbstractNsolt}
    for idx = 1:length(targets)
        filename_nsolt = joinpath(dirname, string("nsolt_", idx, ".json"))
        savefb(filename_nsolt, targets.filterbanks[idx])
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

gettrainingdata(filename::AbstractString) = gettrainingdata(FileIO.load(filename))
gettrainingdata(td::AbstractArray{T}) where {T<:AbstractFloat} = td
gettrainingdata(td::AbstractArray{Complex{T}}) where {T<:AbstractFloat} = td
function gettrainingdata(td::AbstractArray{T,D}, TP::Type=Float64) where {D,T<:Color}
    channelview(td) .|> TP
end

function lossfcn(cb::LearningTarget, x::AbstractArray, y::AbstractArray, params; shape=Shapes.Vec(size(x))) where {TT<:AbstractArray,TM<:AbstractArray}
    cpcb = similar_dictionary(cb, params)
    syn = createTransform(cpcb, shape)
    norm(x - synthesize(syn, y))^2/2
end

function similar_dictionary(nsolt::AbstractNsolt, (θ, μ)::Tuple{TT,TM}) where {TT<:AbstractArray,TM<:AbstractArray}
    setrotations!(similar(nsolt, eltype(θ)), θ, μ)
end

similar_dictionary(target::Multiscale, params::NTuple{N}) where {N} = Multiscale(similar_dictionary.(target.filterbanks, params)...)

getParamsDictionary(nsolt::AbstractNsolt) = getrotations(nsolt)
setParamsDictionary!(nsolt::AbstractNsolt, pm::NTuple{2}) = setrotations!(nsolt, pm...)

getParamsDictionary(targets::Multiscale) = map(t->getParamsDictionary(t), targets.filterbanks)
setParamsDictionary!(targets::Multiscale, pm::NTuple) = map((t,p)->setParamsDictionary!(t, p), targets.filterbanks, pm)

getOperatorType_scs(::Type, ::Val) = AbstractOperator
getOperatorType_scs(::Type{NS}, ::Val{:convolution}) where {NS<:AbstractNsolt} = ConvolutionalOperator

decompose_params(::Type, params) = (params, ())
compose_params(::Type, params, ()) = params

decompose_params(::Type{NS}, (θ, μ)) where {NS<:AbstractNsolt} = (θ, μ)
compose_params(::Type{NS}, θ, μ) where {NS<:AbstractNsolt} = (θ, μ)

function decompose_params(::Type{Multiscale}, params::NTuple{N}) where {N} # T<:AbstractNsolt
    θs, μs = map(t->t[1], params), map(t->t[2], params)
    vcat(θs...), (length.(θs), μs)
end

function compose_params(::Type{Multiscale}, vp::AbstractArray, pminfo::Tuple) where {N}
    arrpms = map(rng->vp[rng],intervals(pminfo[1]))
    map((lhs,rhs)->(lhs,rhs,), arrpms, pminfo[2])
end

initialstate(spr::Optimizers.Steepest, args...) = ()
initialstate(spr::Optimizers.AdaGrad, args...) = 0.0
initialstate(adam::Optimizers.Adam, ∇f::Function, vecpm::AbstractArray) = (zero(vecpm), 0.0)

updateamount(stp::Optimizers.Steepest, grad, args...) = (stp.rate .* grad, ())
function updateamount(agd::Optimizers.AdaGrad, grad::AbstractArray, ::Integer, v::AbstractFloat)
    v = v + norm(grad)^2
    upm = agd.rate / (sqrt(v) + agd.ϵ) * grad
    (upm, v)
end
function updateamount(adam::Optimizers.Adam, grad::AbstractArray, itr::Integer, (m, v)::Tuple{TM,TV}) where {TM<:AbstractArray,TV<:AbstractFloat}
    m = adam.β1 * m + (1 - adam.β1) * grad
    v = adam.β2 * v + (1 - adam.β2) * norm(grad)^2
    m̂ = m / (1 - adam.β1^(itr))
    v̂ = v / (1 - adam.β2^(itr))

    upm = adam.rate * m̂ / (sqrt(v̂) + adam.ϵ)
    (upm, (m, v))
end

getvalidshape(shape::Shapes.AbstractShape, args...) = shape
getvalidshape(::Nothing, ::LearningTarget, sc, du, x::AbstractArray) = Shapes.Vec(size(x))
getvalidshape(::Nothing, ::Multiscale, ::SparseCoders.IHT{T}, args...) where {N,T<:NTuple{N}} = Shapes.Arrayed()

namestring(nsolt::Rnsolt) = namestring(nsolt, "Real NSOLT")
namestring(nsolt::Cnsolt) = namestring(nsolt, "Complex NSOLT")
namestring(nsolt::AbstractNsolt{T,D}, strnsolt::AbstractString) where {T,D} = "$D-dimensional $(ifelse(istype1(nsolt), "Type-I", "Type-II")) $strnsolt"

verbosenames() = Dict(:none => 0, :standard => 1, :specified => 2, :loquacious => 3)
verboselevel(sym::Symbol) = verbosenames()[sym]
verboselevel(lv::Integer) = lv

display_filters(::Val{false}, args...; kwargs...) = nothing

display_filters(::Val{true}, nsolt::AbstractNsolt) = display(plot(nsolt))
